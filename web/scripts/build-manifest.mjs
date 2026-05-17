#!/usr/bin/env node
/**
 * Scans the output/ directory for *.jsonl conversation files and produces
 * a compact manifest.json for the web conversation viewer.
 *
 * Output: web/public/manifest.json
 */

import { readdir, readFile, writeFile, mkdir, copyFile } from 'node:fs/promises';
import { join, resolve } from 'node:path';
import { createReadStream } from 'node:fs';
import { createInterface } from 'node:readline';

const ROOT = resolve(import.meta.dirname, '..', '..');
const OUTPUT_DIR = join(ROOT, 'output');
const PUBLIC_DIR = join(ROOT, 'web', 'public');
const MANIFEST_PATH = join(PUBLIC_DIR, 'manifest.json');

async function readFirstLine(filePath) {
  const stream = createReadStream(filePath, { encoding: 'utf-8' });
  const rl = createInterface({ input: stream, crlfDelay: Infinity });
  try {
    for await (const line of rl) {
      return line;
    }
  } finally {
    rl.close();
    stream.destroy();
  }
  return null;
}

function stripModelPrefix(modelId) {
  const idx = modelId.indexOf('/');
  return idx >= 0 ? modelId.slice(idx + 1) : modelId;
}

async function scanConversation(filePath) {
  const firstLine = await readFirstLine(filePath);
  if (!firstLine) return null;

  let meta;
  try {
    meta = JSON.parse(firstLine);
  } catch {
    return null;
  }
  if (meta.type !== 'metadata') return null;

  const fileName = filePath.split('/').pop();

  return {
    id: meta.conversation_id,
    fileName,
    profile: meta.human_profile?.name ?? 'Unknown',
    profileBackstory: (meta.human_profile?.backstory ?? '').slice(0, 200),
    aiModel: meta.ai_model,
    aiModelShort: stripModelPrefix(meta.ai_model ?? ''),
    humanModel: meta.human_model,
    humanModelShort: stripModelPrefix(meta.human_model ?? ''),
    language: meta.language ?? 'english',
    turns: meta.total_turns ?? 0,
    tokens: meta.total_tokens_estimate ?? 0,
    cost: meta.total_cost_usd ?? 0,
    seedWords: meta.seed_words ?? [],
    startedAt: meta.started_at ?? '',
    finishedAt: meta.finished_at ?? '',
    aiTemperature: meta.ai_temperature,
    humanTemperature: meta.human_temperature,
    aiMaxTokens: meta.ai_max_tokens,
    humanMaxTokens: meta.human_max_tokens,
    companionMode: meta.companion_mode ?? 'honest',
  };
}

async function main() {
  console.log('Building manifest...');

  let entries;
  try {
    entries = await readdir(OUTPUT_DIR);
  } catch {
    console.warn('output/ directory not found, creating empty manifest');
    entries = [];
  }

  const jsonlFiles = entries
    .filter(f => f.endsWith('.jsonl'))
    .map(f => join(OUTPUT_DIR, f));

  console.log(`Scanning ${jsonlFiles.length} JSONL files...`);

  const conversations = [];
  for (const filePath of jsonlFiles) {
    const conv = await scanConversation(filePath);
    if (conv) conversations.push(conv);
  }

  conversations.sort((a, b) => {
    if (a.startedAt && b.startedAt) return b.startedAt.localeCompare(a.startedAt);
    return b.id.localeCompare(a.id);
  });

  const profiles = [...new Set(conversations.map(c => c.profile))].sort();
  const aiModels = [...new Set(conversations.map(c => c.aiModel))].sort();
  const humanModels = [...new Set(conversations.map(c => c.humanModel))].sort();

  const manifest = {
    generatedAt: new Date().toISOString(),
    totalConversations: conversations.length,
    totalTurns: conversations.reduce((s, c) => s + c.turns, 0),
    totalCost: conversations.reduce((s, c) => s + c.cost, 0),
    profiles,
    aiModels,
    humanModels,
    conversations,
  };

  await mkdir(PUBLIC_DIR, { recursive: true });
  await writeFile(MANIFEST_PATH, JSON.stringify(manifest));

  const sizeKb = (Buffer.byteLength(JSON.stringify(manifest)) / 1024).toFixed(1);
  console.log(`Manifest written to ${MANIFEST_PATH} (${sizeKb} KB)`);
  console.log(`  ${conversations.length} conversations, ${manifest.totalTurns} total turns`);

  // Copy JSONL files to public/output/ for static serving in production
  const outputPublic = join(PUBLIC_DIR, 'output');
  await mkdir(outputPublic, { recursive: true });
  let copied = 0;
  for (const file of jsonlFiles) {
    const name = file.split('/').pop();
    await copyFile(file, join(outputPublic, name));
    copied++;
  }
  console.log(`Copied ${copied} JSONL files to ${outputPublic}`);
}

main().catch(e => { console.error(e); process.exit(1); });
