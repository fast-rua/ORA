// Apply a context pack's view (ora-pi/src/context/context_pack.ts) to recorded requests.
//
//   node tests/support/context_pack/view_driver.ts INPUT.json OUTPUT.json
//
// INPUT: {binding, delivery: {messages, tools}, images: {sha256(url): url}, steps: [{id, messages, tools}]}.
// The RPC answers come from INPUT: `context_pack.delivery` is `delivery`, `context_pack.images`
// looks each URL up in `images` (Python downscaled them with the same function the session's
// RPC uses). OUTPUT: {steps: [{id, messages, tools, manifest}]}.

import { createHash } from "node:crypto";
import { readFileSync, writeFileSync } from "node:fs";

import { ContextPackView } from "../../../ora-pi/src/context/context_pack.ts";
import { dumpsPy, parsePy } from "../../../ora-pi/src/pyjson.ts";

type Json = any;

const [inputPath, outputPath] = process.argv.slice(2);
const input: Json = parsePy(readFileSync(inputPath, "utf8"));
const rpc = {
  async request(method: string, params: Json): Promise<Json> {
    if (method === "context_pack.delivery") return input.delivery;
    if (method === "context_pack.images") {
      return {
        images: params.images.map((url: string) => {
          const found = input.images[createHash("sha256").update(url).digest("hex")];
          if (typeof found !== "string") throw new Error("an image was not downscaled by the driver's caller");
          return found;
        }),
      };
    }
    throw new Error(`unexpected RPC ${method}`);
  },
};
const steps: Json[] = [];
for (const step of input.steps) {
  // One view per step: the session's view keeps only its delivery and image caches across requests.
  const view = new ContextPackView(input.binding, rpc);
  const projected = await view.apply({ messages: step.messages, tools: step.tools, manifest: {} });
  steps.push({ id: step.id, messages: projected.messages, tools: projected.tools, manifest: projected.manifest });
}
writeFileSync(outputPath, dumpsPy({ steps }, { ensureAscii: false }));
