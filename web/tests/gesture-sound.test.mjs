import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import vm from "node:vm";
import ts from "typescript";

test("React gesture playback preserves event IDs and a single effect channel", async () => {
  const effects = [];
  const audios = [];
  const updates = [];
  let nextError = null;
  let snapshot = null;
  let poll;
  class AudioStub {
    constructor(url) {
      this.url = url;
      this.paused = true;
      this.ended = false;
      this.listeners = {};
      audios.push(this);
    }
    addEventListener(name, callback) { this.listeners[name] = callback; }
    play() {
      if (nextError) {
        const error = nextError;
        nextError = null;
        return Promise.reject(error);
      }
      this.paused = false;
      return Promise.resolve();
    }
    pause() { this.paused = true; }
    removeAttribute() { this.url = null; }
    load() {}
  }
  const react = {
    useCallback: (fn) => fn,
    useEffect: (fn) => effects.push(fn),
    useRef: (value) => ({ current: value }),
    useState: (value) => [value, (next) => updates.push(next)],
  };
  const api = {
    getState: async () => snapshot,
    getGestures: async () => ["peace", "heart"].map((label) => ({ label, sound: `/sound/${label}` })),
  };
  const source = readFileSync(new URL("../src/hooks/useRobotState.ts", import.meta.url), "utf8");
  const compiled = ts.transpileModule(source, {
    compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2020 },
  }).outputText;
  const context = {
    exports: {},
    require: (name) => {
      if (name === "react") return react;
      if (name === "@/lib/api") return api;
      if (name === "@/lib/audio") return { registerAudio: () => () => {} };
      throw new Error(`Unexpected module: ${name}`);
    },
    Audio: AudioStub,
    DOMException,
    window: { setInterval: (fn) => { poll = fn; return 1; }, clearInterval: () => {} },
  };
  vm.runInNewContext(compiled, context);
  const hook = context.exports.useRobotState({ applyServerState: () => {} });
  const cleanup = effects.map((effect) => effect());
  await new Promise(setImmediate);
  const send = async (gesture, gesture_event) => {
    snapshot = { gesture, gesture_event };
    await poll();
  };
  await send("peace", 1);
  assert.equal(audios.length, 1);
  await send("peace", 1);
  await send("peace", 2);
  assert.equal(audios.length, 1, "same active effect must not restart");
  await send("heart", 3);
  assert.equal(audios.length, 2);
  assert.equal(audios[0].paused, true, "replacement stops the previous audio");
  audios[0].listeners.error();
  assert.equal(audios[1].paused, false, "stale callbacks cannot stop the replacement");
  audios[1].ended = true;
  audios[1].listeners.ended();
  await send("heart", 4);
  assert.equal(audios.length, 3, "a new event replays even if polling missed the release");
  hook.stopGestureSound();
  assert.equal(audios[2].paused, true, "starting a voice turn stops the effect");
  await send("none", undefined);
  await send("peace", undefined);
  assert.equal(audios.length, 4, "older servers still work without event IDs");
  nextError = new DOMException("autoplay blocked", "NotAllowedError");
  hook.previewGestureSound("heart");
  await new Promise(setImmediate);
  assert.equal(updates.at(-1), "Gesture audio blocked by browser.");
  assert.equal(audios[3].paused, true, "preview shares the same effect channel");
  hook.previewGestureSound("heart");
  assert.equal(audios.at(-1).url, "/sound/heart");
  assert.equal(audios.at(-1).paused, false, "manual preview can retry a blocked effect");
  hook.previewGestureSound("stop");
  assert.equal(updates.at(-1), "Sound file not installed for stop.");
  cleanup.forEach((fn) => fn?.());
  assert.equal(audios.at(-1).paused, true, "unmount stops playback");
});
