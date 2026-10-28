// Run with: node --test tests/js/*.test.mjs
import assert from 'node:assert/strict';
import { test } from 'node:test';

import { prepMixin } from '../../static/js/prep.js';

const ok = (data) => ({ ok: true, data });
const fail = (error, status = 502) => ({ ok: false, error, status });

const AREAS = [
    { key: 'architecture', title: 'Architecture', case: 'RAG bot', goal: 'Design it', status: 'not_started', lesson: null, checks: [] },
    { key: 'design', title: 'Design', case: 'API', goal: 'Shape it', status: 'not_started', lesson: null, checks: [] },
];
const PLAN = { id: 'p1', role: 'AI Engineer', seniority: 'senior', has_resume: false, areas: AREAS };
const LESSON = {
    case_title: 'RAG bot', scenario: 'S', design: 'D', trade_offs: 'T', failure_modes: 'F',
    testing_and_observability: 'O', interview_angle: 'I', checks: ['q1', 'q2'],
};

/** A stand-in for the Alpine app: the mixin's own properties (getters kept) plus the app's fields. */
function setup(api = {}, app = {}) {
    const state = Object.defineProperties(
        { screen: 'setup', jobRole: 'AI Engineer', jobDescription: 'jd', resumeText: 'cv', ...app },
        Object.getOwnPropertyDescriptors(prepMixin(api)),
    );
    return state;
}

const deferred = () => {
    let resolve;
    const promise = new Promise((r) => { resolve = r; });
    return { promise, resolve };
};

test('picking Prepare loads the role templates once', async () => {
    let calls = 0;
    const s = setup({ templates: async () => { calls++; return ok({ templates: [{ role: 'AI Architect', focus: 'x' }] }); } });
    await s.setSetupMode('prep');
    await s.setSetupMode('mock');
    await s.setSetupMode('prep');
    assert.equal(s.setupMode, 'prep');
    assert.deepEqual(s.prepTemplates, [{ role: 'AI Architect', focus: 'x' }]);
    assert.equal(calls, 1);
});

test('templates failing to load does not break the setup screen', async () => {
    const s = setup({ templates: async () => fail('nope', 0) });
    await s.setSetupMode('prep');
    assert.equal(s.setupMode, 'prep');
    assert.deepEqual(s.prepTemplates, []);
    assert.equal(s.prepError, '');
});

test('createPrepPlan sends the setup form and opens the plan', async () => {
    let sent;
    const s = setup({ createPlan: async (input) => { sent = input; return ok(PLAN); } });
    s.prepSeniority = 'staff';
    await s.createPrepPlan();
    assert.deepEqual(sent, { role: 'AI Engineer', seniority: 'staff', jobDescription: 'jd', resumeText: 'cv' });
    assert.equal(s.screen, 'prep-plan');
    assert.equal(s.prepPlan.id, 'p1');
    assert.equal(s.prepBusy, '');
    assert.equal(s.prepError, '');
});

test('createPrepPlan does nothing without a role', async () => {
    let called = false;
    const s = setup({ createPlan: async () => { called = true; return ok(PLAN); } }, { jobRole: '   ' });
    await s.createPrepPlan();
    assert.equal(called, false);
    assert.equal(s.screen, 'setup');
});

test('createPrepPlan ignores a second click while one is running', async () => {
    const gate = deferred();
    let calls = 0;
    const s = setup({ createPlan: async () => { calls++; return gate.promise; } });
    const first = s.createPrepPlan();
    assert.match(s.prepBusy, /Building/);
    await s.createPrepPlan();
    gate.resolve(ok(PLAN));
    await first;
    assert.equal(calls, 1);
});

test('a failed createPrepPlan shows the error and can be retried', async () => {
    const replies = [fail("Couldn't generate your prep plan. Please try again."), ok(PLAN)];
    const s = setup({ createPlan: async () => replies.shift() });
    await s.createPrepPlan();
    assert.equal(s.prepError, "Couldn't generate your prep plan. Please try again.");
    assert.equal(s.screen, 'setup');
    assert.equal(s.prepBusy, '');
    await s.retryPrep();
    assert.equal(s.prepError, '');
    assert.equal(s.screen, 'prep-plan');
});

test('opening an Area generates its Lesson once and then reuses it', async () => {
    let opened = 0;
    const s = setup({ openLesson: async (id, key) => { opened++; return ok({ ...AREAS[0], key, status: 'ready', lesson: LESSON }); } });
    s.prepPlan = PLAN;
    await s.openPrepLesson('architecture');
    assert.equal(s.screen, 'prep-lesson');
    assert.equal(s.prepArea.status, 'ready');
    assert.deepEqual(s.prepSections.map((x) => x.key),
        ['scenario', 'design', 'trade_offs', 'failure_modes', 'testing_and_observability', 'interview_angle']);
    assert.equal(s.nextPrepArea.key, 'design');

    s.screen = 'prep-plan';
    await s.openPrepLesson('architecture');
    assert.equal(opened, 1);  // served from what is already loaded
    assert.equal(s.screen, 'prep-lesson');
});

test('opening a Lesson scrolls to the top when the app can', async () => {
    let scrolled = 0;
    const s = setup({ openLesson: async () => ok({ ...AREAS[0], status: 'ready', lesson: LESSON }) },
        { scrollPrepToTop: () => { scrolled++; } });
    s.prepPlan = PLAN;
    await s.openPrepLesson('architecture');
    assert.equal(scrolled, 1);
});

test('a failed Lesson stays on the plan with a retry', async () => {
    const replies = [fail("Couldn't generate this lesson. Please try again."), ok({ ...AREAS[0], status: 'ready', lesson: LESSON })];
    const s = setup({ openLesson: async () => replies.shift() });
    s.prepPlan = PLAN;
    s.screen = 'prep-plan';
    await s.openPrepLesson('architecture');
    assert.equal(s.screen, 'prep-plan');
    assert.match(s.prepError, /lesson/);
    await s.retryPrep();
    assert.equal(s.screen, 'prep-lesson');
});

test('an unknown Area is ignored', async () => {
    const s = setup({ openLesson: async () => { throw new Error('should not be called'); } });
    s.prepPlan = PLAN;
    await s.openPrepLesson('astrology');
    assert.equal(s.screen, 'setup');
});

test('a Lesson that arrives after the user went back is dropped', async () => {
    const gate = deferred();
    const s = setup({ openLesson: async () => gate.promise, listPlans: async () => ok({ plans: [] }) });
    s.prepPlan = PLAN;
    s.screen = 'prep-plan';
    const pending = s.openPrepLesson('architecture');
    assert.match(s.prepBusy, /lesson/);
    await s.showPrepList();  // the user leaves while it is generating
    gate.resolve(ok({ ...AREAS[0], status: 'ready', lesson: LESSON }));
    await pending;
    assert.equal(s.screen, 'prep-list');
    assert.equal(s.prepBusy, '');
    assert.equal(s.prepPlan.areas[0].lesson, null);
});

function lessonOpen(api) {
    const s = setup(api);
    s.prepPlan = { ...PLAN, areas: [{ ...AREAS[0], status: 'ready', lesson: LESSON }, AREAS[1]] };
    s.prepAreaKey = 'architecture';
    s.screen = 'prep-lesson';
    return s;
}

test('answering a Check records feedback and score and clears the draft', async () => {
    let sent;
    const s = lessonOpen({
        answerCheck: async (...args) => { sent = args; return ok({ idx: 1, question: 'q2', answer: 'my answer', score: 7, feedback: 'Good.' }); },
    });
    s.checkDrafts = ['', '  my answer  '];
    await s.submitCheck(1);
    assert.deepEqual(sent, ['p1', 'architecture', 1, 'my answer']);
    assert.equal(s.prepCheckAnswer(1).score, 7);
    assert.equal(s.prepCheckAnswer(0), null);
    assert.deepEqual(s.checkDrafts, ['', '']);
    assert.equal(s.checkBusy, -1);
    assert.equal(s.prepAverage(s.prepArea), 7);
});

test('answering a Check again replaces the earlier answer and keeps order', async () => {
    const replies = [
        ok({ idx: 1, answer: 'b1', score: 4, feedback: 'f' }),
        ok({ idx: 0, answer: 'a1', score: 6, feedback: 'f' }),
        ok({ idx: 1, answer: 'b2', score: 9, feedback: 'f' }),
    ];
    const s = lessonOpen({ answerCheck: async () => replies.shift() });
    for (const [idx, text] of [[1, 'b1'], [0, 'a1'], [1, 'b2']]) {
        s.checkDrafts[idx] = text;
        await s.submitCheck(idx);
    }
    assert.deepEqual(s.prepArea.checks.map((c) => [c.idx, c.answer]), [[0, 'a1'], [1, 'b2']]);
});

test('an unscored Check keeps its feedback', async () => {
    const s = lessonOpen({ answerCheck: async () => ok({ idx: 0, answer: 'a', score: null, feedback: 'Solid.' }) });
    s.checkDrafts[0] = 'a';
    await s.submitCheck(0);
    assert.equal(s.prepCheckAnswer(0).feedback, 'Solid.');
    assert.equal(s.prepAverage(s.prepArea), null);
});

test('an empty draft is not sent, and nor is a second one while busy', async () => {
    const gate = deferred();
    let calls = 0;
    const s = lessonOpen({ answerCheck: async () => { calls++; return gate.promise; } });
    await s.submitCheck(0);
    s.checkDrafts = ['   ', ''];
    await s.submitCheck(0);
    assert.equal(calls, 0);
    s.checkDrafts = ['real', 'other'];
    const first = s.submitCheck(0);
    assert.equal(s.checkBusy, 0);
    await s.submitCheck(1);
    gate.resolve(ok({ idx: 0, answer: 'real', score: 5, feedback: 'f' }));
    await first;
    assert.equal(calls, 1);
});

test('a failed Check keeps the draft, shows the error and can be retried', async () => {
    const replies = [fail("Couldn't generate feedback. Please try again."), ok({ idx: 0, answer: 'a', score: 5, feedback: 'f' })];
    const s = lessonOpen({ answerCheck: async () => replies.shift() });
    s.checkDrafts[0] = 'a';
    await s.submitCheck(0);
    assert.match(s.prepError, /feedback/);
    assert.equal(s.checkDrafts[0], 'a');
    assert.equal(s.checkBusy, -1);
    await s.retryPrep();
    assert.equal(s.prepCheckAnswer(0).score, 5);
    assert.equal(s.prepError, '');
});

test('marking a Lesson done updates its status', async () => {
    const s = lessonOpen({ completeLesson: async () => ok({ ...AREAS[0], status: 'done' }) });
    await s.completePrepLesson();
    assert.equal(s.prepArea.status, 'done');
    assert.equal(s.prepProgress.done, 1);
    assert.equal(s.prepArea.lesson.case_title, 'RAG bot');  // the Lesson text is kept
});

test('going back to the plan clears errors and stale work', async () => {
    const s = lessonOpen({});
    s.prepError = 'old';
    s.prepCanRetry = true;
    s.checkBusy = 1;
    s.backToPrepPlan();
    assert.equal(s.screen, 'prep-plan');
    assert.equal(s.prepError, '');
    assert.equal(s.prepCanRetry, false);
    assert.equal(s.checkBusy, -1);
});

test('a Check reply that arrives after navigating away is dropped', async () => {
    const gate = deferred();
    const s = lessonOpen({ answerCheck: async () => gate.promise });
    s.checkDrafts[0] = 'a';
    const pending = s.submitCheck(0);
    s.backToPrepPlan();
    gate.resolve(ok({ idx: 0, answer: 'a', score: 5, feedback: 'f' }));
    await pending;
    assert.equal(s.prepCheckAnswer(0), null);
});

test('showPrepList loads plans, and reports a failure with a retry', async () => {
    const replies = [fail('server down', 500), ok({ plans: [{ id: 'p1', role: 'SRE' }] })];
    const s = setup({ listPlans: async () => replies.shift() });
    await s.showPrepList();
    assert.equal(s.screen, 'prep-list');
    assert.equal(s.prepListLoading, false);
    assert.equal(s.prepError, 'server down');
    assert.deepEqual(s.prepPlans, []);
    await s.retryPrep();
    assert.deepEqual(s.prepPlans, [{ id: 'p1', role: 'SRE' }]);
    assert.equal(s.prepError, '');
});

test('openPrepPlan loads the saved plan', async () => {
    const s = setup({ getPlan: async (id) => ok({ ...PLAN, id }) });
    await s.openPrepPlan('p9');
    assert.equal(s.screen, 'prep-plan');
    assert.equal(s.prepPlan.id, 'p9');
});

test('progress and the next Area follow the plan', async () => {
    const s = setup({});
    assert.deepEqual(s.prepProgress, { total: 0, done: 0, started: 0 });
    assert.equal(s.prepArea, null);
    assert.equal(s.nextPrepArea, null);
    s.prepPlan = { ...PLAN, areas: [{ ...AREAS[0], status: 'done' }, { ...AREAS[1], status: 'ready' }] };
    s.prepAreaKey = 'design';
    assert.deepEqual(s.prepProgress, { total: 2, done: 1, started: 2 });
    assert.equal(s.nextPrepArea, null);  // last Area
    assert.equal(s.prepStatusLabel('ready'), 'In progress');
});

test('the retry action is never exposed as reactive state', async () => {
    // Alpine calls any function an expression evaluates to, which re-ran the failed action in a loop.
    const s = setup({ createPlan: async () => fail('nope') });
    await s.createPrepPlan();
    assert.equal(s.prepCanRetry, true);
    assert.equal(Object.values(s).some((v) => typeof v === 'function' && v.name === 'retryAction'), false);
    assert.equal('prepRetry' in s, false);
    s.backToPrepPlan();
    assert.equal(s.prepCanRetry, false);
    s.retryPrep();  // nothing to repeat: must not throw or call the API
});

test('retryPrep repeats a failed action exactly once per call', async () => {
    let calls = 0;
    const s = setup({ createPlan: async () => { calls++; return fail('still down'); } });
    await s.createPrepPlan();
    assert.equal(calls, 1);
    await s.retryPrep();
    await Promise.resolve();
    assert.equal(calls, 2);
    assert.equal(s.prepError, 'still down');
    assert.equal(s.prepCanRetry, true);
});
