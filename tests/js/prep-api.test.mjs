// Run with: node --test tests/js/*.test.mjs
import assert from 'node:assert/strict';
import { test } from 'node:test';

import { PrepApi, averageScore, lessonSections, planProgress, statusLabel } from '../../static/js/prep-api.js';

function setup(respond) {
    const calls = [];
    const fetch = async (path, init) => {
        calls.push({ path, ...init, body: init.body ? JSON.parse(init.body) : undefined });
        return respond(path, init);
    };
    return { api: new PrepApi({ fetch }), calls };
}

const json = (status, data) => ({ ok: status >= 200 && status < 300, status, json: async () => data });

test('templates is a GET and returns the data', async () => {
    const { api, calls } = setup(() => json(200, { templates: [] }));
    assert.deepEqual(await api.templates(), { ok: true, data: { templates: [] } });
    assert.equal(calls[0].method, 'GET');
    assert.equal(calls[0].path, '/api/prep/templates');
    assert.equal(calls[0].body, undefined);
});

test('createPlan posts snake_case fields as JSON', async () => {
    const { api, calls } = setup(() => json(201, { id: 'p1' }));
    const result = await api.createPlan({ role: 'AI Engineer', seniority: 'senior', jobDescription: 'jd', resumeText: 'cv' });
    assert.deepEqual(result, { ok: true, data: { id: 'p1' } });
    assert.equal(calls[0].method, 'POST');
    assert.equal(calls[0].path, '/api/prep/plans');
    assert.equal(calls[0].headers['Content-Type'], 'application/json');
    assert.deepEqual(calls[0].body, {
        role: 'AI Engineer', seniority: 'senior', job_description: 'jd', resume_text: 'cv',
    });
});

test('createPlan defaults the optional fields to empty text', async () => {
    const { api, calls } = setup(() => json(201, {}));
    await api.createPlan({ role: 'SRE', seniority: 'mid' });
    assert.equal(calls[0].body.job_description, '');
    assert.equal(calls[0].body.resume_text, '');
});

test('plan, lesson, check and complete paths', async () => {
    const { api, calls } = setup(() => json(200, {}));
    await api.listPlans();
    await api.getPlan('p1');
    await api.openLesson('p1', 'architecture');
    await api.answerCheck('p1', 'architecture', 1, 'my answer');
    await api.completeLesson('p1', 'architecture');
    assert.deepEqual(calls.map((c) => `${c.method} ${c.path}`), [
        'GET /api/prep/plans',
        'GET /api/prep/plans/p1',
        'POST /api/prep/plans/p1/lessons/architecture',
        'POST /api/prep/plans/p1/lessons/architecture/checks/1',
        'POST /api/prep/plans/p1/lessons/architecture/complete',
    ]);
    assert.deepEqual(calls[3].body, { answer: 'my answer' });
});

test('ids and area keys are URL-encoded', async () => {
    const { api, calls } = setup(() => json(200, {}));
    await api.openLesson('a/b', 'x y?');
    assert.equal(calls[0].path, '/api/prep/plans/a%2Fb/lessons/x%20y%3F');
});

test('an error response surfaces the server message and status', async () => {
    const { api } = setup(() => json(502, { error: "Couldn't generate this lesson. Please try again." }));
    assert.deepEqual(await api.openLesson('p1', 'design'), {
        ok: false, error: "Couldn't generate this lesson. Please try again.", status: 502,
    });
});

test('an error response without a usable message gets a generic one', async () => {
    for (const body of [{}, { error: '' }, { error: 5 }, null]) {
        const { api } = setup(() => json(500, body));
        const result = await api.listPlans();
        assert.equal(result.ok, false);
        assert.equal(result.error, 'Something went wrong. Please try again.');
        assert.equal(result.status, 500);
    }
});

test('a non-JSON error body does not throw', async () => {
    const { api } = setup(() => ({ ok: false, status: 500, json: async () => { throw new SyntaxError('bad'); } }));
    const result = await api.listPlans();
    assert.deepEqual(result, { ok: false, error: 'Something went wrong. Please try again.', status: 500 });
});

test('a 200 that is not JSON is treated as a failure', async () => {
    const { api } = setup(() => ({ ok: true, status: 200, json: async () => { throw new SyntaxError('bad'); } }));
    assert.equal((await api.listPlans()).ok, false);
});

test('a network failure is reported, not thrown', async () => {
    const api = new PrepApi({ fetch: async () => { throw new TypeError('Failed to fetch'); } });
    assert.deepEqual(await api.createPlan({ role: 'SRE', seniority: 'mid' }), {
        ok: false, error: "Couldn't reach the server. Check your connection and try again.", status: 0,
    });
});

test('planProgress from a full plan', () => {
    assert.deepEqual(planProgress({ areas: [{ status: 'done' }, { status: 'ready' }, { status: 'not_started' }] }),
        { total: 3, done: 1, started: 2 });
});

test('planProgress from a list item', () => {
    assert.deepEqual(planProgress({ areas_total: 10, lessons_generated: 4, lessons_done: 2 }),
        { total: 10, done: 2, started: 4 });
    assert.deepEqual(planProgress({}), { total: 0, done: 0, started: 0 });
});

test('statusLabel', () => {
    assert.equal(statusLabel('done'), 'Done');
    assert.equal(statusLabel('ready'), 'In progress');
    assert.equal(statusLabel('not_started'), 'Not started');
    assert.equal(statusLabel('mystery'), 'Not started');
});

test('lessonSections come in reading order and skip empty ones', () => {
    const lesson = {
        interview_angle: 'say it well', scenario: 'the problem', design: '1. a', trade_offs: '  ',
        failure_modes: 'it breaks', testing_and_observability: 'measure it', case_title: 'ignored', checks: ['q'],
    };
    assert.deepEqual(lessonSections(lesson).map((s) => s.key),
        ['scenario', 'design', 'failure_modes', 'testing_and_observability', 'interview_angle']);
    assert.equal(lessonSections(lesson)[0].title, 'The scenario');
    assert.deepEqual(lessonSections(null), []);
});

test('averageScore ignores unscored checks', () => {
    assert.equal(averageScore([{ score: 8 }, { score: null }, { score: 6 }]), 7);
    assert.equal(averageScore([{ score: null }]), null);
    assert.equal(averageScore([]), null);
    assert.equal(averageScore(undefined), null);
});
