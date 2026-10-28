/**
 * Browser client for the Prep plan REST API, plus the small pure helpers the
 * Prep screens use. Every call resolves to `{ ok: true, data }` or
 * `{ ok: false, error, status }`; it never throws, so screens can show the
 * message and offer a retry.
 */

const NETWORK_ERROR = "Couldn't reach the server. Check your connection and try again.";
const GENERIC_ERROR = 'Something went wrong. Please try again.';

export class PrepApi {
    constructor({ fetch = (...args) => globalThis.fetch(...args) } = {}) {
        this._fetch = fetch;
    }

    templates() { return this._request('GET', '/api/prep/templates'); }

    createPlan({ role, seniority, jobDescription = '', resumeText = '' }) {
        return this._request('POST', '/api/prep/plans', {
            role, seniority, job_description: jobDescription, resume_text: resumeText,
        });
    }

    listPlans() { return this._request('GET', '/api/prep/plans'); }

    getPlan(planId) { return this._request('GET', `/api/prep/plans/${encodeURIComponent(planId)}`); }

    openLesson(planId, area) { return this._request('POST', this._lessonPath(planId, area)); }

    answerCheck(planId, area, idx, answer) {
        return this._request('POST', `${this._lessonPath(planId, area)}/checks/${encodeURIComponent(idx)}`, { answer });
    }

    completeLesson(planId, area) { return this._request('POST', `${this._lessonPath(planId, area)}/complete`); }

    _lessonPath(planId, area) {
        return `/api/prep/plans/${encodeURIComponent(planId)}/lessons/${encodeURIComponent(area)}`;
    }

    async _request(method, path, body) {
        const init = { method };
        if (body !== undefined) {
            init.headers = { 'Content-Type': 'application/json' };
            init.body = JSON.stringify(body);
        }
        let response;
        try {
            response = await this._fetch(path, init);
        } catch (e) {
            return { ok: false, error: NETWORK_ERROR, status: 0 };
        }
        let data = null;
        try {
            data = await response.json();
        } catch (e) { /* not JSON: handled below */ }
        if (!response.ok) {
            const error = data && typeof data.error === 'string' && data.error ? data.error : GENERIC_ERROR;
            return { ok: false, error, status: response.status };
        }
        if (data === null) return { ok: false, error: GENERIC_ERROR, status: response.status };
        return { ok: true, data };
    }
}

// --- Pure helpers ---

/** Progress through a Prep plan, from a plan (with `areas`) or a list item (with counts). */
export function planProgress(plan) {
    if (Array.isArray(plan.areas)) {
        const total = plan.areas.length;
        const done = plan.areas.filter((a) => a.status === 'done').length;
        const started = plan.areas.filter((a) => a.status !== 'not_started').length;
        return { total, done, started };
    }
    return { total: plan.areas_total || 0, done: plan.lessons_done || 0, started: plan.lessons_generated || 0 };
}

export function statusLabel(status) {
    return { not_started: 'Not started', ready: 'In progress', done: 'Done' }[status] || 'Not started';
}

const SECTIONS = [
    ['scenario', 'The scenario'],
    ['design', 'The design'],
    ['trade_offs', 'Trade-offs'],
    ['failure_modes', 'Failure modes'],
    ['testing_and_observability', 'Testing and observability'],
    ['interview_angle', 'In the interview'],
];

/** A Lesson's markdown sections in reading order, skipping any that are empty. */
export function lessonSections(lesson) {
    if (!lesson) return [];
    return SECTIONS
        .filter(([key]) => typeof lesson[key] === 'string' && lesson[key].trim())
        .map(([key, title]) => ({ key, title, markdown: lesson[key] }));
}

/** Average Score of an Area's answered Checks, or null if none has a Score. */
export function averageScore(checks) {
    const scores = (checks || []).map((c) => c.score).filter((s) => typeof s === 'number');
    if (!scores.length) return null;
    return scores.reduce((a, b) => a + b, 0) / scores.length;
}
