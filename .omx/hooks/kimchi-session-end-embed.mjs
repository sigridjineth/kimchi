import { spawn } from 'node:child_process';
import { existsSync } from 'node:fs';
import { homedir } from 'node:os';
import { join, resolve } from 'node:path';

function text(value, fallback = '') {
  return typeof value === 'string' ? value : fallback;
}

function object(value) {
  return value && typeof value === 'object' ? value : {};
}

function parseExtraArgs(raw) {
  const s = text(raw).trim();
  if (!s) return [];
  return s.split(/\s+/).filter(Boolean);
}

export async function onHookEvent(event, sdk) {
  if (!event || event.event !== 'session-end') return;

  const sessionId = text(event.session_id).trim();
  if (!sessionId) {
    await sdk.log.warn('kimchi hook skipped: missing session_id');
    return;
  }

  const lastSession = text(await sdk.state.read('last_session_id', ''), '');
  if (lastSession === sessionId) {
    return;
  }

  const ctx = object(event.context);
  const projectPath = text(ctx.project_path, process.cwd());
  const scriptPath = resolve(projectPath, 'kimchi.py');

  if (!existsSync(scriptPath)) {
    await sdk.log.warn('kimchi hook skipped: kimchi.py not found', {
      project_path: projectPath,
      script_path: scriptPath,
      session_id: sessionId,
    });
    return;
  }

  const python = text(process.env.KIMCHI_PYTHON, 'python3');
  const kimchiHome = text(process.env.KIMCHI_HOME, join(homedir(), '.kimchi'));
  const kimchiCell = text(process.env.KIMCHI_CELL, 'codex_code');
  const sessionRoot = text(process.env.KIMCHI_SESSION_ROOT, join(homedir(), '.codex', 'sessions'));
  const extraArgs = parseExtraArgs(process.env.KIMCHI_HOOK_EXTRA_ARGS);

  const args = [
    scriptPath,
    'index',
    '--home', kimchiHome,
    '--cell', kimchiCell,
    '--session-root', sessionRoot,
    '--session-id', sessionId,
    ...extraArgs,
  ];

  const child = spawn(python, args, {
    cwd: projectPath,
    detached: true,
    stdio: 'ignore',
    env: process.env,
  });
  child.unref();

  await sdk.state.write('last_session_id', sessionId);
  await sdk.state.write('last_launch', {
    at: new Date().toISOString(),
    session_id: sessionId,
    project_path: projectPath,
    command: [python, ...args],
  });

  await sdk.log.info('kimchi hook queued session embedding', {
    session_id: sessionId,
    project_path: projectPath,
    kimchi_home: kimchiHome,
    session_root: sessionRoot,
  });
}
