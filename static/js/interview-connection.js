/**
 * Interview connection: the browser side of one Interview session.
 *
 * Owns the WebSocket, the saved session id (so a refresh can resume), the
 * reconnect rules, and the translation of server events into the state the
 * screens show. It never touches the DOM: every change calls `onChange` with a
 * plain snapshot, and collaborators are injected so tests can drive it by hand.
 */

const STORAGE_KEY = 'activeInterview';
const MAX_RECONNECT_ATTEMPTS = 6;
const MAX_RECONNECT_DELAY_MS = 15000;

const capitalize = (s) => (s ? s.charAt(0).toUpperCase() + s.slice(1) : '');

function defaultStorage() {
    try { return globalThis.sessionStorage; } catch (e) { return null; }
}

export class InterviewConnection {
    constructor({
        onChange,
        createSocket = (url) => new WebSocket(url),
        storage = defaultStorage(),
        setTimeout: setTimer = (fn, ms) => globalThis.setTimeout(fn, ms),
        clearTimeout: clearTimer = (id) => globalThis.clearTimeout(id),
        location = globalThis.location,
    } = {}) {
        this._onChange = onChange || (() => {});
        this._createSocket = createSocket;
        this._storage = storage;
        this._setTimer = setTimer;
        this._clearTimer = clearTimer;
        this._location = location;

        this._ws = null;
        this._allowReconnect = false;
        this._reconnectAttempts = 0;
        this._reconnectTimer = null;
        this._active = { jobRole: '', interviewType: 'mixed' };
        this._clear();
    }

    // --- Interface ---

    start({ job_position, job_description = '', interview_type = 'mixed', resume_text = '' }) {
        this._teardown();
        this._clear();
        this._active = { jobRole: job_position, interviewType: interview_type };
        this._s.screen = 'chat';
        this._open({ job_position, job_description, interview_type, resume_text });
        this._emit();
    }

    /** Reconnect to the session saved by an earlier page load. Returns the saved record, or null. */
    resumeIfActive() {
        const active = this._loadActive();
        if (!active) return null;
        this._teardown();
        this._clear();
        this._s.sessionId = active.sessionId;
        this._active = { jobRole: active.jobRole || '', interviewType: active.interviewType || 'mixed' };
        this._s.screen = 'chat';
        this._open({ type: 'resume', session_id: active.sessionId });
        this._emit();
        return { jobRole: this._active.jobRole, interviewType: this._active.interviewType, sessionId: active.sessionId };
    }

    /** Send the candidate's answer. Returns false if it could not be sent now. */
    send(text) {
        text = (text || '').trim();
        if (!text || !this._s.inputEnabled) return false;
        this._s.messages.push({ type: 'user', content: text });
        this._ws.send(JSON.stringify({ type: 'user_input', content: text }));
        this._s.inputEnabled = false;
        this._s.questionsAnswered++;
        this._emit();
        return true;
    }

    /** End the interview now, or queue the end until the Turn in flight finishes. */
    end() {
        if (this._s.questionsAnswered < 1) return;
        if (this._s.inputEnabled) {
            this._sendEnd();
        } else {
            this._s.pendingEnd = true;
            this._s.messages.push({ type: 'system', content: 'Will end after current round finishes...' });
        }
        this._emit();
    }

    /** Drop the connection and the saved session, back to a blank setup screen. */
    reset(setupNotice = '') {
        this._teardown();
        this._clearActive();
        this._clear();
        this._s.setupNotice = setupNotice;
        this._emit();
    }

    // --- Internals ---

    _clear() {
        this._s = {
            messages: [],
            inputEnabled: false,
            isTyping: false,
            typingSource: 'Thinking...',
            summaryData: null,
            sessionId: null,
            questionsAnswered: 0,
            totalQuestions: 0,
            pendingEnd: false,
            connected: false,
            reconnecting: false,
            screen: 'setup',
            setupNotice: '',
        };
    }

    _emit() {
        this._onChange({ ...this._s, messages: this._s.messages.slice() });
    }

    _teardown() {
        this._allowReconnect = false;
        this._reconnectAttempts = 0;
        this._clearTimer(this._reconnectTimer);
        this._reconnectTimer = null;
        if (this._ws) {
            const ws = this._ws;
            this._ws = null;  // so its onclose does not try to reconnect
            ws.close();
        }
    }

    _open(firstMessage) {
        const protocol = this._location.protocol === 'https:' ? 'wss' : 'ws';
        const ws = this._createSocket(`${protocol}://${this._location.host}/ws/interview`);
        this._ws = ws;
        this._allowReconnect = true;

        ws.onopen = () => {
            this._s.connected = true;
            this._s.reconnecting = false;
            this._reconnectAttempts = 0;
            ws.send(JSON.stringify(firstMessage));
            this._emit();
        };
        ws.onmessage = (event) => {
            this._handle(JSON.parse(event.data));
            this._emit();
        };
        ws.onclose = () => {
            if (this._ws !== ws) return;  // replaced or reset
            this._s.connected = false;
            this._s.isTyping = false;
            this._s.inputEnabled = false;
            this._scheduleReconnect();
            this._emit();
        };
    }

    _scheduleReconnect() {
        const s = this._s;
        if (!this._allowReconnect || s.screen !== 'chat') return;
        if (!s.sessionId) {
            // The connection dropped before the server created a session
            this._allowReconnect = false;
            s.messages.push({ type: 'system', content: 'Connection error. Please refresh and try again.' });
            return;
        }
        if (this._reconnectAttempts >= MAX_RECONNECT_ATTEMPTS) {
            this._allowReconnect = false;
            s.reconnecting = false;
            s.messages.push({ type: 'system', content: 'Connection lost. Refresh the page to try again.' });
            return;
        }
        const delay = Math.min(1000 * 2 ** this._reconnectAttempts, MAX_RECONNECT_DELAY_MS);
        this._reconnectAttempts++;
        s.reconnecting = true;
        this._reconnectTimer = this._setTimer(
            () => this._open({ type: 'resume', session_id: s.sessionId }),
            delay,
        );
    }

    _handle(data) {
        const s = this._s;
        switch (data.type) {
            case 'agent_message':
                s.isTyping = false;
                s.messages.push({
                    type: 'agent',
                    source: data.source,
                    sourceName: capitalize(data.source),
                    content: data.content,
                    metadata: data.metadata || {},
                });
                break;

            case 'system_event':
                this._handleSystemEvent(data);
                break;

            case 'summary':
                s.isTyping = false;
                this._allowReconnect = false;
                this._clearActive();
                s.summaryData = data.metadata || {};
                s.screen = 'summary';
                break;

            case 'error':
                s.isTyping = false;
                if (data.code === 'session_not_found') {
                    this._teardown();
                    this._clearActive();
                    this._clear();
                    this._s.setupNotice = data.content;
                    break;
                }
                if (data.fatal) {
                    this._allowReconnect = false;
                    this._clearActive();
                }
                s.messages.push({ type: 'system', content: 'Error: ' + data.content });
                break;
        }
    }

    _handleSystemEvent(data) {
        const s = this._s;
        const meta = data.metadata || {};
        switch (data.event) {
            case 'interview_started':
                if (meta.num_questions) s.totalQuestions = meta.num_questions;
                if (meta.session_id) {
                    s.sessionId = meta.session_id;
                    this._saveActive();
                }
                s.messages.push({ type: 'system', content: data.content });
                break;

            case 'resumed':
                s.sessionId = meta.session_id || s.sessionId;
                s.totalQuestions = meta.num_questions || s.totalQuestions;
                s.questionsAnswered = meta.questions_answered || 0;
                s.messages = (meta.messages || []).map((m) => (m.type === 'user'
                    ? { type: 'user', content: m.content }
                    : {
                        type: 'agent',
                        source: m.source,
                        sourceName: capitalize(m.source),
                        content: m.content,
                        metadata: m.metadata || {},
                    }));
                s.messages.push({ type: 'system', content: 'Reconnected. Continuing your interview.' });
                s.screen = 'chat';
                break;

            case 'agent_typing':
                s.isTyping = true;
                s.typingSource = 'Thinking...';
                break;

            case 'waiting_for_input':
                s.isTyping = false;
                s.inputEnabled = true;
                if (s.pendingEnd) this._sendEnd();  // the candidate already asked to end
                break;

            case 'interview_complete':
                s.isTyping = false;
                s.inputEnabled = false;
                break;
        }
    }

    _sendEnd() {
        const s = this._s;
        s.messages.push({ type: 'system', content: 'Ending interview and generating scorecard...' });
        this._ws.send(JSON.stringify({ type: 'end_interview' }));
        s.inputEnabled = false;
        s.pendingEnd = false;
        s.isTyping = true;
        s.typingSource = 'Generating scorecard...';
    }

    // The session id lives in sessionStorage so a refresh can pick the interview back up
    _saveActive() {
        try {
            this._storage.setItem(STORAGE_KEY, JSON.stringify({
                sessionId: this._s.sessionId,
                jobRole: this._active.jobRole,
                interviewType: this._active.interviewType,
            }));
        } catch (e) { /* storage unavailable: resume across refresh is skipped */ }
    }

    _loadActive() {
        try {
            const raw = this._storage.getItem(STORAGE_KEY);
            const active = raw ? JSON.parse(raw) : null;
            return active && active.sessionId ? active : null;
        } catch (e) {
            return null;
        }
    }

    _clearActive() {
        try { this._storage.removeItem(STORAGE_KEY); } catch (e) { /* ignore */ }
    }
}
