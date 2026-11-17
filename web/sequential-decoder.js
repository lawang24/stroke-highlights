import {Input, BlobSource, ALL_FORMATS, VideoSampleSink, EncodedPacketSink, Mp4InputFormat} from 'mediabunny';
export const abortError = () => new DOMException('Cancelled', 'AbortError');
export function checkAbort(signal) { if (signal.aborted) throw abortError(); }
// Chromium's HTMLMediaElement normalizes through MediaTimeForTimeValue(), then
// WebMediaPlayerImpl::Seek() converts that double to TimeDelta again. Both
// conversions truncate toward zero. The intermediate division must remain: a
// mathematically equivalent single truncation changes IEEE-754 boundary cases.
export const chromiumSeekMicroseconds = time => Math.trunc(Math.trunc(time * 1e6) / 1e6 * 1e6);
export const atOrBefore = (pts, time) => Math.round(pts * 1e6) <= chromiumSeekMicroseconds(time);
// FFmpeg av_rescale_q rounds positive seek microseconds to the stream timebase.
// Exact integer arithmetic avoids introducing a third floating-point rounding.
export const seekTick = (time, resolution) => Number((BigInt(chromiumSeekMicroseconds(time)) * BigInt(resolution) + 500000n) / 1000000n);
export const selectable = (pts, time, resolution, isKey) => atOrBefore(pts,time) || (isKey && Math.round(pts*resolution) <= seekTick(time,resolution));

export async function createSequentialSource(blob, signal, stats = {}) {
  checkAbort(signal);
  if (typeof VideoDecoder === 'undefined') throw new Error('WebCodecs unavailable');
  const input = new Input({source: new BlobSource(blob, {maxCacheSize: 8 * 1024 ** 2}), formats: ALL_FORMATS});
  let iterator, held, next, closed = false;
  Object.assign(stats, {decoded: 0, maxHeldSamples: 0, closed: false});
  const close = async () => {
    if (closed) return;
    closed = true;
    signal.removeEventListener('abort', onAbort);
    // return() wakes a pending next() before disposing the demuxer.
    await iterator?.return();
    held?.close(); next?.close(); held = next = null;
    input.dispose(); stats.closed = true;
  };
  const onAbort = () => { void close(); };
  signal.addEventListener('abort', onAbort, {once: true});
  try {
    if (!(await input.getFormat() instanceof Mp4InputFormat)) throw new Error('Unverified container: preserve baseline decoding with fallback');
    const track = await input.getPrimaryVideoTrack();
    checkAbort(signal);
    if (!track || !await track.canDecode()) throw new Error('No WebCodecs decoder for this track');
    const audio = await input.getPrimaryAudioTrack();
    if (await track.getCodec() !== 'avc' || (audio && await audio.getCodec() !== 'aac') || await track.getFirstTimestamp() !== 0) {
      throw new Error('Unverified codec or timeline: preserve baseline decoding with fallback');
    }
    const resolution = await track.getTimeResolution();
    if (!Number.isSafeInteger(resolution) || resolution <= 0) throw new Error('Unverified timebase: preserve baseline decoding with fallback');
    const keyTicks = new Set();
    stats.timeResolution = resolution;
    const config = await track.getDecoderConfig();
    const color = config?.colorSpace;
    if (!color?.primaries || !color?.matrix || !color?.transfer || typeof color.fullRange !== 'boolean') {
      throw new Error('Incomplete color metadata: preserve video-element rendering with fallback');
    }
    // Native seeking produces different decoded pixels within some held VFR
    // frames. Until that decoder path is equivalent, retain baseline rendering.
    let frameDuration;
    stats.preflightPackets = 0;
    for await (const packet of new EncodedPacketSink(track).packets(undefined, undefined, {metadataOnly: true})) {
      checkAbort(signal);
      stats.preflightPackets++;
      if (packet.type === 'key') keyTicks.add(Math.round(packet.timestamp * resolution));
      if (!(packet.duration > 0) || (frameDuration !== undefined && Math.abs(packet.duration - frameDuration) > 0.0000005)) {
        throw new Error('Variable or unknown frame duration: preserve video-element rendering with fallback');
      }
      frameDuration ??= packet.duration;
    }
    stats.keyframes = keyTicks.size;
    iterator = new VideoSampleSink(track).samples();
    const read = async () => {
      checkAbort(signal);
      const {value, done} = await iterator.next();
      if (signal.aborted || closed) { value?.close(); throw abortError(); }
      if (!done) stats.decoded++;
      return done ? null : value;
    };
    next = await read();
    return {
      stats, close,
      async draw(ctx, time, width, height) {
        checkAbort(signal);
        while (next && selectable(next.timestamp, time, resolution, keyTicks.has(Math.round(next.timestamp * resolution)))) {
          held?.close(); held = next; next = null;
          next = await read();
          stats.maxHeldSamples = Math.max(stats.maxHeldSamples, Number(!!held) + Number(!!next));
        }
        if (!held) throw new Error('No frame at requested source time');
        held.draw(ctx, 0, 0, width, height);
        return {pts: held.timestamp, duration: held.duration};
      },
    };
  } catch (error) { await close(); throw signal.aborted ? abortError() : error; }
}
