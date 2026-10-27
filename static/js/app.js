import { InterviewConnection } from './interview-connection.js';

window.interviewApp = function interviewApp() {
    // Not reactive on purpose: Alpine would wrap the connection in a Proxy.
    let connection = null;
    let lastConnectionScreen = null;

    return {
        // State
        screen: 'setup',       // 'setup' | 'chat' | 'summary' | 'history'
        jobRole: '',
        jobDescription: '',
        interviewType: 'mixed',
        interviewTypes: [
            { value: 'mixed', label: 'Mixed', icon: '\u{1F500}', desc: 'Behavioral + Technical' },
            { value: 'behavioral', label: 'Behavioral', icon: '\u{1F4AC}', desc: 'Past experience & soft skills' },
            { value: 'technical', label: 'Technical', icon: '\u{1F4BB}', desc: 'Coding & problem-solving' },
            { value: 'system_design', label: 'System Design', icon: '\u{1F3D7}', desc: 'Architecture & scale' },
        ],
        messages: [],
        userInput: '',
        inputEnabled: false,
        isTyping: false,
        typingSource: 'Thinking...',
        summaryData: null,
        questionsAnswered: 0,
        totalQuestions: 0,
        pendingEnd: false,
        connected: false,
        reconnecting: false,
        copyBtnText: 'Copy to Clipboard',
        resumeText: '',
        resumeFilename: '',
        resumeChars: 0,
        resumeUploading: false,
        resumeError: '',
        historyItems: [],
        historyLoading: false,
        sessionId: null,
        viewStatus: '',        // status of a past interview opened from history
        setupNotice: '',

        get statusLabel() {
            if (this.screen === 'summary') {
                if (this.viewStatus === 'abandoned') return 'Abandoned';
                if (this.viewStatus === 'in_progress') return 'Unfinished';
                return 'Completed';
            }
            if (this.reconnecting) return 'Reconnecting\u2026';
            if (this.connected) return 'Live';
            return 'Disconnected';
        },
        get statusClass() {
            if (this.screen === 'summary') {
                return this.viewStatus === 'completed' || !this.viewStatus ? 'status-done' : 'status-disconnected';
            }
            if (this.reconnecting) return 'status-reconnecting';
            if (this.connected) return 'status-live';
            return 'status-disconnected';
        },
        get isUnfinishedView() {
            return this.screen === 'summary' && !this.summaryData;
        },
        get hasCategoryScores() {
            return !!(this.summaryData && this.summaryData.scores && Object.keys(this.summaryData.scores).length);
        },
        get overallScore() {
            if (!this.summaryData || !this.summaryData.scores) return 0;
            const scores = Object.values(this.summaryData.scores);
            if (scores.length === 0) return 0;
            return scores.reduce((a, b) => a + b, 0) / scores.length;
        },

        // The Interview connection owns the socket, reconnects and resume; it
        // reports every change as a snapshot, and the DOM effects happen here.
        init() {
            connection = new InterviewConnection({ onChange: (snapshot) => this.applyConnection(snapshot) });
            const active = connection.resumeIfActive();
            if (active) {
                this.jobRole = active.jobRole;
                this.interviewType = active.interviewType;
            }
        },

        applyConnection(snapshot) {
            const before = {
                messages: this.messages.length,
                typing: this.isTyping,
                enabled: this.inputEnabled,
                screen: this.screen,
            };
            // The screen is only taken from the connection when it changed, so a
            // late socket event cannot pull the user back from History.
            const { screen, ...rest } = snapshot;
            Object.assign(this, rest);
            if (screen !== lastConnectionScreen) {
                lastConnectionScreen = screen;
                this.screen = screen;
            }
            if (snapshot.messages.length !== before.messages || (snapshot.isTyping && !before.typing)
                || this.screen !== before.screen) {
                this.scrollToBottom();
            }
            if (snapshot.inputEnabled && !before.enabled) {
                this.$nextTick(() => {
                    if (this.$refs.answerInput) this.$refs.answerInput.focus();
                });
            }
        },

        // Actions
        startInterview() {
            if (!this.jobRole.trim()) return;

            this.viewStatus = '';
            this.setupNotice = '';
            connection.start({
                job_position: this.jobRole,
                job_description: this.jobDescription,
                interview_type: this.interviewType,
                resume_text: this.resumeText,
            });
        },

        sendAnswer() {
            if (!connection.send(this.userInput)) return;
            this.userInput = '';
            // Reset textarea height
            if (this.$refs.answerInput) {
                this.$refs.answerInput.style.height = 'auto';
            }
        },

        endInterview() {
            connection.end();
        },

        resetInterview() {
            connection.reset();
            this.viewStatus = '';
            this.userInput = '';
            this.copyBtnText = 'Copy to Clipboard';
            this.resumeText = '';
            this.resumeFilename = '';
            this.resumeChars = 0;
            this.resumeError = '';
        },

        copyScorecard() {
            if (!this.summaryData) return;

            const s = this.summaryData;
            let text = `Interview Scorecard - ${this.jobRole}\n`;
            text += '='.repeat(40) + '\n\n';

            if (s.scores) {
                text += 'SCORES\n';
                for (const [cat, score] of Object.entries(s.scores)) {
                    text += `  ${this.formatCategory(cat)}: ${score}/10\n`;
                }
                text += '\n';
            }

            if (s.strengths && s.strengths.length) {
                text += 'STRENGTHS\n';
                s.strengths.forEach(str => text += `  - ${str}\n`);
                text += '\n';
            }

            if (s.improvements && s.improvements.length) {
                text += 'AREAS FOR IMPROVEMENT\n';
                s.improvements.forEach(imp => text += `  - ${imp}\n`);
                text += '\n';
            }

            if (s.overall_summary) {
                text += 'OVERALL\n';
                text += `  ${s.overall_summary}\n`;
            }

            navigator.clipboard.writeText(text).then(() => {
                this.copyBtnText = 'Copied!';
                setTimeout(() => { this.copyBtnText = 'Copy to Clipboard'; }, 2000);
            });
        },

        // History
        async showHistory() {
            this.screen = 'history';
            this.historyLoading = true;
            try {
                const res = await fetch('/api/interviews?limit=20');
                const data = await res.json();
                this.historyItems = data.interviews || [];
            } catch (e) {
                this.historyItems = [];
            }
            this.historyLoading = false;
        },

        async viewHistoryDetail(id) {
            try {
                const res = await fetch(`/api/interviews/${id}`);
                if (!res.ok) return;
                const interview = await res.json();
                this.jobRole = interview.job_position;
                this.interviewType = interview.interview_type || 'mixed';
                this.viewStatus = interview.status || 'completed';
                this.summaryData = interview.summary_data || null;
                this.messages = (interview.messages || [])
                    .filter(m => m.name !== 'summary')  // the scorecard replaces the raw JSON
                    .map(m => ({
                        type: m.role === 'user' ? 'user' : 'agent',
                        source: m.name || 'interviewer',
                        sourceName: this.capitalize(m.name || 'interviewer'),
                        content: m.content,
                        metadata: {},
                    }));
                this.screen = 'summary';
            } catch (e) {
                // silently fail
            }
        },

        // Resume upload
        async uploadResume(event) {
            const file = event.target.files[0];
            if (!file) return;

            this.resumeError = '';
            this.resumeUploading = true;

            const formData = new FormData();
            formData.append('file', file);

            try {
                const res = await fetch('/api/upload-resume', {
                    method: 'POST',
                    body: formData,
                });
                const data = await res.json();
                if (!res.ok || data.error) {
                    this.resumeError = data.error || 'Failed to upload resume. Please try again.';
                } else {
                    this.resumeText = data.text;
                    this.resumeFilename = data.filename;
                    this.resumeChars = data.chars;
                }
            } catch (e) {
                this.resumeError = 'Failed to upload resume. Please try again.';
            }
            this.resumeUploading = false;
        },

        removeResume() {
            this.resumeText = '';
            this.resumeFilename = '';
            this.resumeChars = 0;
            this.resumeError = '';
        },

        // Helpers
        autoResize(event) {
            const el = event.target;
            el.style.height = 'auto';
            el.style.height = Math.min(el.scrollHeight, 120) + 'px';
        },

        renderMarkdown(text) {
            if (!text) return '';
            if (typeof marked !== 'undefined' && typeof DOMPurify !== 'undefined') {
                return DOMPurify.sanitize(marked.parse(text));
            }
            return text;
        },

        scrollToBottom() {
            this.$nextTick(() => {
                const el = this.$refs.messages;
                if (el) el.scrollTop = el.scrollHeight;
            });
        },

        capitalize(s) {
            if (!s) return '';
            return s.charAt(0).toUpperCase() + s.slice(1);
        },

        formatCategory(cat) {
            return cat.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
        },

        scoreColor(score) {
            if (score >= 7) return 'score-high';
            if (score >= 5) return 'score-mid';
            return 'score-low';
        },

        gradeColor(score) {
            if (score >= 7) return '#10b981';
            if (score >= 5) return '#f59e0b';
            return '#ef4444';
        },

        gradeVerdict(score) {
            if (score >= 9) return 'Outstanding performance';
            if (score >= 8) return 'Strong candidate';
            if (score >= 7) return 'Above average performance';
            if (score >= 6) return 'Solid foundation, room to grow';
            if (score >= 5) return 'Average performance';
            if (score >= 4) return 'Below expectations';
            return 'Needs significant improvement';
        },

        // Check if this is the very first interviewer message (no separator needed)
        isFirstInterviewer(index) {
            for (let i = 0; i < index; i++) {
                if (this.messages[i].type === 'agent' && this.messages[i].source === 'interviewer') {
                    return false;
                }
            }
            return true;
        },

        // Get the round number for an interviewer message at a given index
        getRoundNumber(index) {
            let round = 0;
            for (let i = 0; i <= index; i++) {
                if (this.messages[i].type === 'agent' && this.messages[i].source === 'interviewer') {
                    round++;
                }
            }
            return round;
        },
    };
};
