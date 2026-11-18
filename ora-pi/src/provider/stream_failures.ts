// Port of ora/agent/providers/stream_failures.py: how a broken provider stream
// and an HTTP error answer are classified for retry.

import { parsePy } from "../pyjson.ts";
import { PolicySessionError, attr, pyStrip } from "./py.ts";
import { transientTransport } from "./recovery.ts";

/** Fresh requests granted to one logical call after breaks that carried content. */
export const STREAM_BREAK_RETRIES = 3;

export const GATEWAY_UPSTREAM_STREAM_BREAK = "stream disconnected before completion";
const LOOPBACK_HOSTS = new Set(["127.0.0.1", "localhost", "::1"]);

/** `urlsplit(url).hostname`: lowercased, IPv6 brackets removed, or null. */
export function urlHostname(url: string): string | null {
  try {
    const hostname = new URL(url).hostname.toLowerCase();
    if (!hostname) return null;
    return hostname.startsWith("[") && hostname.endsWith("]") ? hostname.slice(1, -1) : hostname;
  } catch {
    return null;
  }
}

/** A loopback gateway's HTTP 408 reporting that its upstream stream broke. */
export function gatewayUpstreamStreamBreak(status: number, detail: string, baseUrl: string): boolean {
  const hostname = urlHostname(baseUrl);
  if (status !== 408 || hostname === null || !LOOPBACK_HOSTS.has(hostname)) return false;
  let parsed: any;
  try {
    parsed = parsePy(detail);
  } catch {
    return false;
  }
  if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) return false;
  const error = Object.prototype.hasOwnProperty.call(parsed, "error") ? parsed.error : undefined;
  const message = typeof error === "object" && error !== null && !Array.isArray(error) ? error.message : undefined;
  return typeof message === "string" && message.includes(GATEWAY_UPSTREAM_STREAM_BREAK);
}

/** One HTTP error answer, classified for a same-request retry. */
export function httpStatusFailure(status: number, detail: string, baseUrl: string): PolicySessionError {
  const failure = new PolicySessionError(pyStrip(`native provider HTTP ${status}: ${detail}`));
  failure.httpStatus = status;
  failure.upstreamStreamBreak = gatewayUpstreamStreamBreak(status, detail, baseUrl);
  failure.retryable = status === 429 || (status >= 500 && status < 600) || failure.upstreamStreamBreak;
  return failure;
}

/** A transport-phase failure after response content had arrived. */
export function isStreamBreak(exc: unknown): boolean {
  const partial = attr(exc, "partialEvents", null);
  return Boolean(attr(exc, "responseStarted", false)
    || (Array.isArray(partial) ? partial.length > 0 : partial)
    || attr(exc, "upstreamStreamBreak", false) === true);
}

/** Carry a broken stream's verdict past the wrapper that reports it (legacy path). */
export function streamFailureAfterContent(cause: unknown): PolicySessionError {
  const retryable = transientTransport(cause);
  const failure = new PolicySessionError(
    "native provider stream failed after response content; "
      + (retryable ? "retryable transport fault" : "not retried"),
    { cause },
  );
  failure.retryable = retryable;
  failure.responseStarted = true;
  return failure;
}
