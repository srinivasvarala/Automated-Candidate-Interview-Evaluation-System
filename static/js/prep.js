/**
 * Prep plan screens: state and actions, as a mixin for the Alpine app.
 *
 * Merge it with `Object.defineProperties(app, Object.getOwnPropertyDescriptors(prepMixin()))`
 * so the getters survive. It expects the app to provide `screen`, `jobRole`,
 * `jobDescription` and `resumeText`, and may call the optional `scrollPrepToTop()`.
 * The server does all the work; this only keeps what the screens show.
 */

import { PrepApi, averageScore, lessonSections, planProgress, statusLabel } from './prep-api.js';

export function prepMixin(api = new PrepApi()) {
    // Kept out of the reactive state on purpose: Alpine calls any function an
    // expression evaluates to, so `x-show="prepRetry"` would run the retry.
    let retryAction = null;

    return {
        // State
        setupMode: 'mock',          // 'mock' | 'prep' (which form the setup screen shows)
        prepSeniority: 'mid',
        prepTemplates: [],
        prepPlans: [],
        prepListLoading: false,
        prepPlan: null,
        prepAreaKey: '',
        prepBusy: '',               // what is being generated right now, or ''
        prepError: '',
        prepCanRetry: false,        // whether retryPrep() has an action to repeat
        prepRun: 0,                 // bumped on every navigation, so a late reply is ignored
        checkDrafts: ['', ''],
        checkBusy: -1,

        get prepArea() {
            return this.prepPlan ? this.prepPlan.areas.find((a) => a.key === this.prepAreaKey) || null : null;
        },
        get prepProgress() {
            return this.prepPlan ? planProgress(this.prepPlan) : { total: 0, done: 0, started: 0 };
        },
        get prepSections() {
            const area = this.prepArea;
            return area ? lessonSections(area.lesson) : [];
        },
        get nextPrepArea() {
            if (!this.prepPlan) return null;
            const i = this.prepPlan.areas.findIndex((a) => a.key === this.prepAreaKey);
            return i >= 0 ? this.prepPlan.areas[i + 1] || null : null;
        },

        prepStatusLabel(status) { return statusLabel(status); },
        prepProgressOf(plan) { return planProgress(plan); },
        prepAverage(area) { return averageScore(area && area.checks); },
        prepCheckAnswer(idx) {
            const area = this.prepArea;
            return (area && area.checks.find((c) => c.idx === idx)) || null;
        },

        // Setup screen
        async setSetupMode(mode) {
            this.setupMode = mode;
            this.prepError = '';
            if (mode === 'prep' && !this.prepTemplates.length) {
                const result = await api.templates();
                if (result.ok) this.prepTemplates = result.data.templates || [];
            }
        },

        async createPrepPlan() {
            const role = this.jobRole.trim();
            if (!role || this.prepBusy) return;
            await this._prepRun(
                'Building your prep plan…',
                () => api.createPlan({
                    role,
                    seniority: this.prepSeniority,
                    jobDescription: this.jobDescription,
                    resumeText: this.resumeText,
                }),
                (plan) => { this.prepPlan = plan; this.screen = 'prep-plan'; },
                () => this.createPrepPlan(),
            );
        },

        // Navigation
        async showPrepList() {
            this._prepLeave();
            this.screen = 'prep-list';
            this.prepListLoading = true;
            const run = this.prepRun;
            const result = await api.listPlans();
            if (run !== this.prepRun) return;
            this.prepListLoading = false;
            if (result.ok) {
                this.prepPlans = result.data.plans || [];
            } else {
                this.prepPlans = [];
                this.prepError = result.error;
                this._setRetry(() => this.showPrepList());
            }
        },

        async openPrepPlan(id) {
            if (this.prepBusy) return;
            await this._prepRun(
                'Opening your plan…',
                () => api.getPlan(id),
                (plan) => { this.prepPlan = plan; this.screen = 'prep-plan'; },
                () => this.openPrepPlan(id),
            );
        },

        backToPrepPlan() {
            this._prepLeave();
            this.screen = 'prep-plan';
        },

        async openPrepLesson(key) {
            if (this.prepBusy || !this.prepPlan) return;
            const area = this.prepPlan.areas.find((a) => a.key === key);
            if (!area) return;
            if (area.lesson) {
                this._showLesson(key);
                return;
            }
            const planId = this.prepPlan.id;
            await this._prepRun(
                'Writing your lesson… this can take up to a minute.',
                () => api.openLesson(planId, key),
                (opened) => { this._replaceArea(opened); this._showLesson(key); },
                () => this.openPrepLesson(key),
            );
        },

        async submitCheck(idx) {
            const draft = (this.checkDrafts[idx] || '').trim();
            const area = this.prepArea;
            if (!draft || this.checkBusy !== -1 || !area) return;
            const planId = this.prepPlan.id;
            const run = this.prepRun;
            this.prepError = '';
            this.checkBusy = idx;
            const result = await api.answerCheck(planId, area.key, idx, draft);
            if (run !== this.prepRun) return;
            this.checkBusy = -1;
            if (!result.ok) {
                this.prepError = result.error;
                this._setRetry(() => this.submitCheck(idx));
                return;
            }
            const answered = this.prepArea;
            const checks = answered.checks.filter((c) => c.idx !== idx).concat(result.data)
                .sort((a, b) => a.idx - b.idx);
            this._replaceArea({ ...answered, checks });
            this.checkDrafts = this.checkDrafts.map((d, i) => (i === idx ? '' : d));
        },

        async completePrepLesson() {
            const area = this.prepArea;
            if (!area || this.prepBusy) return;
            const planId = this.prepPlan.id;
            await this._prepRun(
                'Saving…',
                () => api.completeLesson(planId, area.key),
                (done) => this._replaceArea({ ...area, status: done.status }),
                () => this.completePrepLesson(),
            );
        },

        retryPrep() {
            const retry = retryAction;
            this.prepError = '';
            this._setRetry(null);
            if (retry) retry();
        },

        // Internals
        _prepLeave() {
            this.prepRun++;  // a reply still in flight is now stale
            this.prepBusy = '';
            this.prepError = '';
            this._setRetry(null);
            this.checkBusy = -1;
        },

        _setRetry(action) {
            retryAction = action;
            this.prepCanRetry = !!action;
        },

        async _prepRun(label, call, onOk, retry) {
            this.prepError = '';
            this._setRetry(null);
            const run = ++this.prepRun;
            this.prepBusy = label;
            const result = await call();
            if (run !== this.prepRun) return;  // the user navigated away meanwhile
            this.prepBusy = '';
            if (!result.ok) {
                this.prepError = result.error;
                this._setRetry(retry);
                return;
            }
            onOk(result.data);
        },

        _replaceArea(area) {
            this.prepPlan = {
                ...this.prepPlan,
                areas: this.prepPlan.areas.map((a) => (a.key === area.key ? area : a)),
            };
        },

        _showLesson(key) {
            this.prepAreaKey = key;
            this.checkDrafts = ['', ''];
            this.screen = 'prep-lesson';
            if (this.scrollPrepToTop) this.scrollPrepToTop();
        },
    };
}
