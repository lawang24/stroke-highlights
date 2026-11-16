from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import tempfile
from threading import Thread
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from regression_review import Dataset, ROOT, digest, handler_for, propose_label, validate_review, write_json


class RegressionReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name); (self.root/'clips').mkdir()
        source = self.root/'source.mp4'; source.write_bytes(b'original')
        clip = self.root/'clips/clip-001.mp4'; clip.write_bytes(b'0123456789')
        self.clip = {'id':'clip-001','file':'clips/clip-001.mp4','sha256':digest(clip),
                     'duration':4,'fps':30,'width':1920,'height':1080,
                     'source_id':'source-1','source_start':10,'source_end':14,
                     'selection':'low_motion_negative_candidate','selection_evidence':{'prior_verdict':'no_shot'}}
        write_json(self.root/'manifest.json', {'id':'test','clips':[self.clip],
            'sources':[{'id':'source-1','path':str(source),'name':source.name,'sha256':digest(source)}]})
        write_json(self.root/'annotations.json', {'version':2,'dataset_id':'test','revision':0,'reviews':{}})
        write_json(self.root/'proposals.json', {'dataset_id':'test','backend_sha256':digest(ROOT/'tennis_rnn_backend.py'),
            'clips':{'clip-001':{'label':'forehand','reason':'Current detector','clip_sha256':self.clip['sha256']}}})
        self.dataset = Dataset(self.root)

    def test_suggestions_never_become_truth_without_human_decision(self):
        self.assertEqual(self.dataset.proposals()['clip-001']['label'], 'forehand')
        with self.assertRaisesRegex(ValueError,'Every review clip'): self.dataset.freeze(0)
        self.assertEqual(list(self.root.glob('frozen*')), [])
        self.assertEqual(self.dataset.annotations()['reviews'], {})

    def test_approve_and_correct_record_explicit_decisions(self):
        with self.assertRaisesRegex(ValueError,'Suggestion changed'):
            self.dataset.save('clip-001',{'label':'backhand','action':'approve'},0)
        approved=self.dataset.save('clip-001',{'label':'forehand','action':'approve'},0)
        self.assertEqual(approved['reviews']['clip-001']['decision'],'approved')
        corrected=self.dataset.save('clip-001',{'label':'backhand','action':'correct'},1)
        self.assertEqual(corrected['reviews']['clip-001']['decision'],'corrected')
        self.assertEqual(corrected['reviews']['clip-001']['label'],'backhand')
        with self.assertRaisesRegex(ValueError,'another tab'):
            self.dataset.save('clip-001',{'label':'no_shot'},0)

    def test_four_labels_need_no_contact_time_or_notes(self):
        for label in ['forehand','backhand','no_shot','unsure']:
            result=validate_review({'label':label},complete=True)
            self.assertEqual(result['label'],label)
            self.assertNotIn('shots',result)
        for label in ['serve','other','shots','']:
            with self.assertRaises(ValueError):validate_review({'label':label})
        with self.assertRaises(ValueError):validate_review({'label':None},complete=True)

    def test_freeze_scores_labels_only_and_preserves_snapshot(self):
        self.dataset.save('clip-001',{'label':'backhand'},0)
        frozen=self.dataset.freeze(1); path=self.root/frozen['file']; original=path.read_bytes()
        snapshot=json.loads(original)
        truth=snapshot['ground_truth'][0]
        self.assertEqual(truth['label'],'backhand');self.assertTrue(truth['included'])
        self.assertNotIn('source_shots',truth);self.assertNotIn('shots',truth)
        self.assertFalse(snapshot['evaluation']['check_detection_times'])
        self.assertFalse(snapshot['evaluation']['check_contact_times'])
        self.assertEqual(snapshot['evaluation']['id'], 'forehand-backhand-positives-v1')
        self.assertEqual(snapshot['evaluation']['truth_aliases']['unsure'], 'no_shot')
        self.assertEqual(snapshot['evaluation']['acceptance']['forehand_false_negatives'], 0)
        self.assertEqual(digest(path),frozen['sha256'])
        self.dataset.save('clip-001',{'label':'unsure'},1)
        self.assertEqual(path.read_bytes(),original)
        with self.assertRaisesRegex(ValueError,'At least one'):self.dataset.freeze(2)

    def test_current_freeze_includes_unsure_as_negative_without_relabeling(self):
        manifest = json.loads((self.root/'manifest.json').read_text())
        manifest['clips'].append({**self.clip, 'id': 'clip-002'})
        write_json(self.root/'manifest.json', manifest)
        dataset = Dataset(self.root)
        dataset.save('clip-001', {'label': 'backhand'}, 0)
        dataset.save('clip-002', {'label': 'unsure'}, 1)
        frozen = dataset.freeze(2)
        snapshot = json.loads((self.root/frozen['file']).read_text())
        uncertain = next(row for row in snapshot['ground_truth'] if row['clip_id'] == 'clip-002')
        self.assertEqual(uncertain['label'], 'unsure')
        self.assertTrue(uncertain['included'])
        self.assertEqual(frozen['included'], 2)
        self.assertEqual(dataset.annotations()['reviews']['clip-002']['label'], 'unsure')

    def test_old_human_label_is_preserved_with_original_backup(self):
        legacy={'version':1,'dataset_id':'test','revision':4,'reviews':{'clip-001':
            {'verdict':'shots','shots':[{'label':'backhand','time':3.999}],'reviewed_at':'earlier'}}}
        write_json(self.root/'annotations.json',legacy)
        migrated=Dataset(self.root).annotations()
        self.assertEqual(migrated['reviews']['clip-001']['label'],'backhand')
        self.assertEqual(migrated['revision'],5)
        self.assertEqual(json.loads((self.root/'annotations-before-simple-review.json').read_text()),legacy)

    def test_proposal_aggregation_handles_tracking_failure_and_mixed_clips(self):
        self.assertEqual(propose_label([],0)['label'],'unsure')
        self.assertEqual(propose_label([],90)['label'],'no_shot')
        self.assertEqual(propose_label([{'stroke':'forehand'}]*2,90)['label'],'forehand')
        self.assertEqual(propose_label([{'stroke':'backhand'},{'stroke':'forehand'}],90)['label'],'unsure')
        self.assertEqual(propose_label([{'stroke':'unknown'}],90)['label'],'unsure')

    def test_modified_video_blocks_freeze_and_stale_proposal_is_hidden(self):
        self.dataset.save('clip-001',{'label':'no_shot'},0)
        (self.root/'clips/clip-001.mp4').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'Clip changed'):self.dataset.freeze(1)
        p=json.loads((self.root/'proposals.json').read_text());p['backend_sha256']='old'
        write_json(self.root/'proposals.json',p)
        self.assertEqual(self.dataset.proposals(),{})

    def test_api_exposes_suggestions_but_hides_sampling_and_limits_files(self):
        server=ThreadingHTTPServer(('127.0.0.1',0),handler_for(self.dataset))
        thread=Thread(target=server.serve_forever,daemon=True);thread.start()
        base=f'http://127.0.0.1:{server.server_port}'
        try:
            with urlopen(base+'/api/dataset') as response:public=json.load(response)
            self.assertNotIn('selection',public['clips'][0])
            self.assertEqual(public['proposals']['clip-001']['label'],'forehand')
            self.assertEqual(public['annotations']['reviews'],{})
            with urlopen(Request(base+'/clips/clip-001.mp4',headers={'Range':'bytes=3-6'})) as response:
                self.assertEqual(response.read(),b'3456')
            payload=json.dumps({'clip_id':'clip-001','revision':0,'review':{'label':'no_shot','action':'correct'}}).encode()
            with self.assertRaises(HTTPError) as error:
                urlopen(Request(base+'/api/review',data=payload,headers={'Content-Type':'application/json','Origin':'https://unrelated.example'}))
            self.assertEqual(error.exception.code,403)
            with urlopen(Request(base+'/api/review',data=payload,headers={'Content-Type':'application/json','Origin':base})) as response:
                self.assertEqual(json.load(response)['revision'],1)
            self.assertEqual(Dataset(self.root).annotations()['reviews']['clip-001']['label'],'no_shot')
            for path in ['/manifest.json','/../pyproject.toml','/clips/../../source.mp4','/source.mp4']:
                with self.assertRaises(HTTPError) as error:urlopen(base+path)
                self.assertEqual(error.exception.code,404)
        finally:server.shutdown();server.server_close();thread.join()


if __name__ == '__main__':unittest.main()
