// Run with: node --test tests/js/*.test.mjs
import assert from 'node:assert/strict';
import { test } from 'node:test';

import { InterviewConnection } from '../../static/js/interview-connection.js';

class FakeSocket {
    constructor(url) {
        this.url = url;
        this.sent = [];
        this.closed = false;
    }
    send(data) { this.sent.push(JSON.parse(data)); }
    close() { this.closed = true; }
    // test drivers
    open() { this.onopen(); }
    receive(data) { this.onmessage({ data: JSON.stringify(data) }); }
    drop() { this.onclose(); }
}

class FakeStorage {
    constructor() { this.data = new Map(); }
    getItem(k) { return this.data.has(k) ? this.data.get(k) : null; }
    setItem(k, v) { this.data.set(k, v); }
    removeItem(k) { this.data.delete(k); }
}

function setup({ storage = new FakeStorage(), protocol = 'http:' } = {}) {
    const sockets = [];
    const timers = [];
    const cleared = [];
    const snapshots = [];
    const connection = new InterviewConnection({
        onChange: (s) => snapshots.push(s),
        createSocket: (url) => { const s = new FakeSocket(url); sockets.push(s); return s; },
        storage,
        setTimeout: (fn, ms) => { timers.push({ fn, ms }); return timers.length; },
        clearTimeout: (id) => cleared.push(id),
        location: { protocol, host: 'localhost:8000' },
    });
    return {
        connection, sockets, timers, cleared, storage,
        get state() { return snapshots.at(-1); },
        get socket() { return sockets.at(-1); },
    };
}

const event = (name, metadata = {}, content = '') => ({ type: 'system_event', event: name, content, metadata });
const interviewer = (content) => ({ type: 'agent_message', source: 'interviewer', content, metadata: {} });

/** Start an interview and get it to the first question, waiting for input. */
function startedAndWaiting(t = setup()) {
    t.connection.start({ job_position: 'Engineer', interview_type: 'technical' });
    t.socket.open();
    t.socket.receive(event('interview_started', { num_questions: 5, session_id: 's1' }, 'Starting...'));
    t.socket.receive(event('agent_typing'));
    t.socket.receive(interviewer('Q1?'));
    t.socket.receive(event('waiting_for_input'));
    return t;
}

test('start opens a socket, sends the setup, and shows the chat screen', () => {
    const t = setup();
    t.connection.start({ job_position: 'Engineer', job_description: 'jd', interview_type: 'technical', resume_text: 'cv' });
    assert.equal(t.socket.url, 'ws://localhost:8000/ws/interview');
    assert.equal(t.state.screen, 'chat');
    t.socket.open();
    assert.deepEqual(t.socket.sent[0], {
        job_position: 'Engineer', job_description: 'jd', interview_type: 'technical', resume_text: 'cv',
    });
    assert.equal(t.state.connected, true);
});

test('uses wss on https pages', () => {
    const t = setup({ protocol: 'https:' });
    t.connection.start({ job_position: 'Engineer' });
    assert.equal(t.socket.url, 'wss://localhost:8000/ws/interview');
});

test('a normal turn: question, answer, feedback', () => {
    const t = startedAndWaiting();
    assert.equal(t.state.sessionId, 's1');
    assert.equal(t.state.totalQuestions, 5);
    assert.equal(t.state.inputEnabled, true);
    assert.equal(t.state.isTyping, false);
    assert.deepEqual(t.state.messages.map((m) => [m.type, m.content]), [['system', 'Starting...'], ['agent', 'Q1?']]);
    assert.equal(t.state.messages[1].sourceName, 'Interviewer');

    assert.equal(t.connection.send('  my answer '), true);
    assert.deepEqual(t.socket.sent.at(-1), { type: 'user_input', content: 'my answer' });
    assert.equal(t.state.inputEnabled, false);
    assert.equal(t.state.questionsAnswered, 1);
    assert.deepEqual(t.state.messages.at(-1), { type: 'user', content: 'my answer' });
});

test('send is refused while not waiting for input, or when empty', () => {
    const t = startedAndWaiting();
    assert.equal(t.connection.send('   '), false);
    t.connection.send('first');
    assert.equal(t.connection.send('second'), false);
    assert.equal(t.socket.sent.filter((m) => m.type === 'user_input').length, 1);
});

test('snapshots do not share the messages array', () => {
    const t = startedAndWaiting();
    const before = t.state.messages;
    t.connection.send('x');
    assert.notEqual(t.state.messages, before);
    assert.equal(before.length, 2);
});

test('end is ignored before the first answer', () => {
    const t = startedAndWaiting();
    t.connection.end();
    assert.equal(t.socket.sent.some((m) => m.type === 'end_interview'), false);
    assert.equal(t.state.pendingEnd, false);
});

test('end sends immediately when waiting for input', () => {
    const t = startedAndWaiting();
    t.connection.send('a');
    t.socket.receive(event('waiting_for_input'));
    t.connection.end();
    assert.deepEqual(t.socket.sent.at(-1), { type: 'end_interview' });
    assert.equal(t.state.isTyping, true);
    assert.equal(t.state.typingSource, 'Generating scorecard...');
    assert.equal(t.state.inputEnabled, false);
});

test('end during a turn is queued and fires at the next waiting_for_input', () => {
    const t = startedAndWaiting();
    t.connection.send('a');
    t.connection.end();
    assert.equal(t.state.pendingEnd, true);
    assert.equal(t.state.messages.at(-1).content, 'Will end after current round finishes...');
    assert.equal(t.socket.sent.some((m) => m.type === 'end_interview'), false);

    t.socket.receive(event('waiting_for_input'));
    assert.deepEqual(t.socket.sent.at(-1), { type: 'end_interview' });
    assert.equal(t.state.pendingEnd, false);
    assert.equal(t.state.inputEnabled, false);
});

test('summary shows the scorecard and forgets the saved session', () => {
    const t = startedAndWaiting();
    assert.ok(t.storage.getItem('activeInterview'));
    t.socket.receive({ type: 'summary', content: '{}', metadata: { scores: { technical: 7 } } });
    assert.equal(t.state.screen, 'summary');
    assert.deepEqual(t.state.summaryData, { scores: { technical: 7 } });
    assert.equal(t.storage.getItem('activeInterview'), null);

    t.socket.drop();  // the socket closing afterwards must not reconnect
    assert.equal(t.timers.length, 0);
});

test('the session id is saved on start with the role and type', () => {
    const t = startedAndWaiting();
    assert.deepEqual(JSON.parse(t.storage.getItem('activeInterview')), {
        sessionId: 's1', jobRole: 'Engineer', interviewType: 'technical',
    });
});

test('resumeIfActive returns null with nothing saved', () => {
    const t = setup();
    assert.equal(t.connection.resumeIfActive(), null);
    assert.equal(t.sockets.length, 0);
});

test('resumeIfActive reconnects with the saved session and replays history', () => {
    const storage = new FakeStorage();
    storage.setItem('activeInterview', JSON.stringify({ sessionId: 's9', jobRole: 'SRE', interviewType: 'behavioral' }));
    const t = setup({ storage });
    assert.deepEqual(t.connection.resumeIfActive(), { sessionId: 's9', jobRole: 'SRE', interviewType: 'behavioral' });
    assert.equal(t.state.screen, 'chat');
    t.socket.open();
    assert.deepEqual(t.socket.sent[0], { type: 'resume', session_id: 's9' });

    t.socket.receive(event('resumed', {
        session_id: 's9', num_questions: 4, questions_answered: 1,
        messages: [
            { type: 'agent', source: 'interviewer', content: 'Q1?', metadata: {} },
            { type: 'user', content: 'A1' },
            { type: 'agent', source: 'evaluator', content: 'FB', metadata: { score: 7, tier: 'Warm-up', tier_index: 0 } },
        ],
    }));
    assert.equal(t.state.questionsAnswered, 1);
    assert.equal(t.state.totalQuestions, 4);
    assert.deepEqual(t.state.messages.map((m) => m.type), ['agent', 'user', 'agent', 'system']);
    assert.equal(t.state.messages[2].metadata.score, 7);
    assert.equal(t.state.messages[3].content, 'Reconnected. Continuing your interview.');
});

test('storage that throws does not break the interview', () => {
    const broken = {
        getItem() { throw new Error('denied'); },
        setItem() { throw new Error('denied'); },
        removeItem() { throw new Error('denied'); },
    };
    const t = setup({ storage: broken });
    assert.equal(t.connection.resumeIfActive(), null);
    startedAndWaiting(t);
    assert.equal(t.state.sessionId, 's1');
    t.socket.receive({ type: 'summary', metadata: {} });
    assert.equal(t.state.screen, 'summary');
});

test('a dropped socket reconnects with 1s, 2s, 4s... backoff and resumes', () => {
    const t = startedAndWaiting();
    t.socket.drop();
    assert.equal(t.state.connected, false);
    assert.equal(t.state.inputEnabled, false);
    assert.equal(t.state.reconnecting, true);
    assert.deepEqual(t.timers.map((x) => x.ms), [1000]);

    t.timers[0].fn();
    assert.equal(t.sockets.length, 2);
    t.socket.drop();  // the retry never connected
    assert.deepEqual(t.timers.map((x) => x.ms), [1000, 2000]);

    t.timers[1].fn();
    t.socket.open();
    assert.deepEqual(t.socket.sent[0], { type: 'resume', session_id: 's1' });
    assert.equal(t.state.reconnecting, false);
    assert.equal(t.state.connected, true);

    t.socket.drop();  // attempts reset after a successful connect
    assert.equal(t.timers.at(-1).ms, 1000);
});

test('backoff is capped at 15s and gives up after six attempts', () => {
    const t = startedAndWaiting();
    t.socket.drop();
    for (let i = 0; i < 5; i++) { t.timers.at(-1).fn(); t.socket.drop(); }
    assert.deepEqual(t.timers.map((x) => x.ms), [1000, 2000, 4000, 8000, 15000, 15000]);
    assert.equal(t.state.reconnecting, true);

    t.timers.at(-1).fn();
    t.socket.drop();
    assert.equal(t.timers.length, 6);
    assert.equal(t.state.reconnecting, false);
    assert.equal(t.state.messages.at(-1).content, 'Connection lost. Refresh the page to try again.');
});

test('a drop before the server created a session does not retry', () => {
    const t = setup();
    t.connection.start({ job_position: 'Engineer' });
    t.socket.open();
    t.socket.drop();
    assert.equal(t.timers.length, 0);
    assert.equal(t.state.messages.at(-1).content, 'Connection error. Please refresh and try again.');
});

test('session_not_found returns to setup with the notice and forgets the session', () => {
    const storage = new FakeStorage();
    storage.setItem('activeInterview', JSON.stringify({ sessionId: 'gone' }));
    const t = setup({ storage });
    t.connection.resumeIfActive();
    t.socket.open();
    t.socket.receive({ type: 'error', code: 'session_not_found', content: 'That interview has expired.' });
    assert.equal(t.state.screen, 'setup');
    assert.equal(t.state.setupNotice, 'That interview has expired.');
    assert.equal(t.state.messages.length, 0);
    assert.equal(storage.getItem('activeInterview'), null);
    assert.equal(t.sockets[0].closed, true);

    t.sockets[0].drop();  // closing the old socket must not reconnect
    assert.equal(t.timers.length, 0);
});

test('a fatal error stops reconnecting and forgets the session', () => {
    const t = startedAndWaiting();
    t.socket.receive({ type: 'error', fatal: true, content: 'Something went wrong.' });
    assert.equal(t.state.messages.at(-1).content, 'Error: Something went wrong.');
    assert.equal(t.storage.getItem('activeInterview'), null);
    t.socket.drop();
    assert.equal(t.timers.length, 0);
});

test('a non-fatal error is shown and the interview continues', () => {
    const t = startedAndWaiting();
    t.socket.receive({ type: 'error', content: 'Please provide an answer.' });
    assert.equal(t.state.messages.at(-1).content, 'Error: Please provide an answer.');
    assert.equal(t.storage.getItem('activeInterview') !== null, true);
    t.socket.drop();
    assert.equal(t.timers.length, 1);
});

test('reset closes the socket, cancels a pending retry and clears everything', () => {
    const t = startedAndWaiting();
    t.socket.drop();
    const first = t.sockets[0];
    t.connection.reset();
    assert.equal(t.cleared.length > 0, true);
    assert.equal(t.state.screen, 'setup');
    assert.deepEqual(t.state.messages, []);
    assert.equal(t.state.sessionId, null);
    assert.equal(t.state.reconnecting, false);
    assert.equal(t.storage.getItem('activeInterview'), null);

    first.drop();
    assert.equal(t.timers.length, 1);  // only the retry scheduled before the reset
});

test('starting again replaces the previous socket', () => {
    const t = startedAndWaiting();
    t.connection.start({ job_position: 'Designer' });
    assert.equal(t.sockets[0].closed, true);
    assert.equal(t.sockets.length, 2);
    assert.deepEqual(t.state.messages, []);
    assert.equal(t.state.sessionId, null);
});

test('interview_complete turns the input off', () => {
    const t = startedAndWaiting();
    t.socket.receive(event('interview_complete'));
    assert.equal(t.state.inputEnabled, false);
    assert.equal(t.state.isTyping, false);
});

test('start with a prep plan sends its id, and without one sends nothing extra', () => {
    const t = setup();
    t.connection.start({ job_position: 'AI Architect', interview_type: 'technical', prep_plan_id: 'plan-1' });
    t.socket.open();
    assert.deepEqual(t.socket.sent[0], {
        job_position: 'AI Architect', job_description: '', interview_type: 'technical', resume_text: '', prep_plan_id: 'plan-1',
    });

    const plain = setup();
    plain.connection.start({ job_position: 'Engineer' });
    plain.socket.open();
    assert.equal('prep_plan_id' in plain.socket.sent[0], false);
});

test('a missing prep plan returns to setup with the notice and starts no reconnects', () => {
    const t = setup();
    t.connection.start({ job_position: 'Engineer', prep_plan_id: 'gone' });
    t.socket.open();
    t.socket.receive({ type: 'error', code: 'prep_plan_not_found', fatal: true, content: 'That prep plan no longer exists.' });
    assert.equal(t.state.screen, 'setup');
    assert.equal(t.state.setupNotice, 'That prep plan no longer exists.');
    assert.equal(t.state.messages.length, 0);
    t.sockets[0].drop();
    assert.equal(t.timers.length, 0);
});
