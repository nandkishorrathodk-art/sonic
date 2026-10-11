/**
 * AVCS (Artificial Visual Cortex System) TypeScript / Node.js Client.
 *
 * Connects to the local AVCS daemon via zero-copy memory-mapped file IPC
 * for sub-millisecond perception streaming in JavaScript and TypeScript environments.
 */

import * as fs from 'fs';
import * as path from 'path';
import * as os from 'os';

export interface AVCSPoint {
  x: number;
  y: number;
}

export interface AVCSPerceptionState {
  sequenceNumber: number;
  timestamp: number;
  latencyMs: number;
  fps: number;
  gazeFixation: AVCSPoint;
  primaryCategory: string;
  surpriseScore: number;
  stability: string;
  affordancesCount: number;
  data: Record<string, any>;
}

export class AVCSNodeClient {
  private bufferPath: string;
  private fd: number | null = null;
  private lastSeq: number = -1;

  constructor(bufferName: string = 'avcs_perception_mmap') {
    this.bufferPath = path.join(os.tmpdir(), `${bufferName}.bin`);
  }

  public connect(): boolean {
    if (!fs.existsSync(this.bufferPath)) {
      return false;
    }
    if (this.fd !== null) {
      try {
        fs.closeSync(this.fd);
      } catch {
        // ignore
      }
      this.fd = null;
    }
    try {
      this.fd = fs.openSync(this.bufferPath, 'r');
      return true;
    } catch {
      this.fd = null;
      return false;
    }
  }

  public readLatest(): AVCSPerceptionState | null {
    if (this.fd === null) {
      if (!this.connect()) {
        return null;
      }
    }

    try {
      const headerBuf = Buffer.alloc(36);
      const headerBytesRead = fs.readSync(this.fd!, headerBuf, 0, 36, 0);
      if (headerBytesRead < 36) {
        return null;
      }

      const magic = headerBuf.subarray(0, 4).toString('utf-8');
      if (magic !== 'AVCS') {
        return null;
      }

      const seq = Number(headerBuf.readBigUInt64LE(12));
      if (seq === this.lastSeq) {
        return null; // No new frame available
      }

      const timestamp = headerBuf.readDoubleLE(20);
      const payloadLen = headerBuf.readUInt32LE(28);
      if (payloadLen <= 0 || payloadLen > 10 * 1024 * 1024) {
        return null;
      }

      const payloadBuf = Buffer.alloc(payloadLen);
      const payloadBytesRead = fs.readSync(this.fd!, payloadBuf, 0, payloadLen, 36);
      if (payloadBytesRead < payloadLen) {
        return null;
      }

      const parsed = JSON.parse(payloadBuf.toString('utf-8'));
      this.lastSeq = seq;

      return {
        sequenceNumber: seq,
        timestamp,
        latencyMs: parsed.latency_ms || 0.0,
        fps: parsed.fps || 0.0,
        gazeFixation: {
          x: parsed.gaze_fixation ? parsed.gaze_fixation[0] : 0.5,
          y: parsed.gaze_fixation ? parsed.gaze_fixation[1] : 0.5,
        },
        primaryCategory: parsed.primary_category || 'unknown',
        surpriseScore: parsed.surprise_score || 0.0,
        stability: parsed.stability || 'unknown',
        affordancesCount: parsed.affordances_count || 0,
        data: parsed,
      };
    } catch {
      return null;
    }
  }

  public close(): void {
    if (this.fd !== null) {
      fs.closeSync(this.fd);
      this.fd = null;
    }
  }
}

