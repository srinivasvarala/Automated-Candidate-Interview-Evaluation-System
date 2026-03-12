function interviewApp() {
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
        ws: null,
        questionsAnswered: 0,
        totalQuestions: 0,
        pendingEnd: false,
        connected: false,
        copyBtnText: 'Copy to Clipboard',
        resumeText: '',
        resumeFilename: '',
        resumeChars: 0,
        resumeUploading: false,
        resumeError: '',
        historyItems: [],
        historyLoading: false,

        get statusLabel() {
            if (this.screen === 'summary') return 'Completed';
            if (this.connected) return 'Live';
            return 'Disconnected';
        },
        get statusClass() {
            if (this.screen === 'summary') return 'status-done';
            if (this.connected) return 'status-live';
            return 'status-disconnected';
        },
        get overallScore() {
            if (!this.summaryData || !this.summaryData.scores) return 0;
            const scores = Object.values(this.summaryData.scores);
            if (scores.length === 0) return 0;
            return scores.reduce((a, b) => a + b, 0) / scores.length;
        },

        // Actions
        startInterview() {
            if (!this.jobRole.trim()) return;

            this.screen = 'chat';
            this.messages = [];
            this.summaryData = null;

            const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
            this.ws = new WebSocket(`${protocol}://${window.location.host}/ws/interview`);

            this.ws.onopen = () => {
                this.connected = true;
                this.ws.send(JSON.stringify({
                    job_position: this.jobRole,
                    job_description: this.jobDescription,
                    interview_type: this.interviewType,
                    resume_text: this.resumeText,
                }));
            };

            this.ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                this.handleMessage(data);
            };

            this.ws.onclose = () => {
                this.connected = false;
                this.isTyping = false;
                this.inputEnabled = false;
            };

            this.ws.onerror = () => {
                this.messages.push({
                    type: 'system',
                    content: 'Connection error. Please refresh and try again.',
                });
            };
        },

        handleMessage(data) {
            switch (data.type) {
                case 'agent_message':
                    this.isTyping = false;
                    this.messages.push({
                        type: 'agent',
                        source: data.source,
                        sourceName: this.capitalize(data.source),
                        content: data.content,
                        metadata: data.metadata || {},
                    });
                    this.scrollToBottom();
                    break;

                case 'system_event':
                    this.handleSystemEvent(data);
                    break;

                case 'summary':
                    this.isTyping = false;
                    this.summaryData = data.metadata || {};
                    this.screen = 'summary';
                    this.scrollToBottom();
                    break;

                case 'error':
                    this.isTyping = false;
                    this.messages.push({
                        type: 'system',
                        content: 'Error: ' + data.content,
                    });
                    this.scrollToBottom();
                    break;
            }
        },

        handleSystemEvent(data) {
            switch (data.event) {
                case 'interview_started':
                    if (data.metadata && data.metadata.num_questions) {
                        this.totalQuestions = data.metadata.num_questions;
                    }
                    this.messages.push({
                        type: 'system',
                        content: data.content,
                    });
                    break;

                case 'agent_typing':
                    this.isTyping = true;
                    this.typingSource = 'Thinking...';
                    this.scrollToBottom();
                    break;

                case 'waiting_for_input':
                    this.isTyping = false;
                    // If user already clicked End, send it now
                    if (this.pendingEnd) {
                        this.inputEnabled = true;
                        this._sendEnd();
                    } else {
                        this.inputEnabled = true;
                        this.$nextTick(() => {
                            if (this.$refs.answerInput) {
                                this.$refs.answerInput.focus();
                            }
                        });
                    }
                    break;

                case 'interview_complete':
                    this.isTyping = false;
                    this.inputEnabled = false;
                    break;
            }
        },

        sendAnswer() {
            const text = this.userInput.trim();
            if (!text || !this.inputEnabled) return;

            this.messages.push({
                type: 'user',
                content: text,
            });

            this.ws.send(JSON.stringify({
                type: 'user_input',
                content: text,
            }));

            this.userInput = '';
            this.inputEnabled = false;
            this.questionsAnswered++;
            // Reset textarea height
            if (this.$refs.answerInput) {
                this.$refs.answerInput.style.height = 'auto';
            }
            this.scrollToBottom();
        },

        endInterview() {
            if (this.questionsAnswered < 1) return;

            // If input is enabled, send end signal immediately
            if (this.inputEnabled) {
                this._sendEnd();
                return;
            }

            // Otherwise queue it — will fire when next waiting_for_input arrives
            this.pendingEnd = true;
            this.messages.push({
                type: 'system',
                content: 'Will end after current round finishes...',
            });
            this.scrollToBottom();
        },

        _sendEnd() {
            this.messages.push({
                type: 'system',
                content: 'Ending interview and generating scorecard...',
            });

            this.ws.send(JSON.stringify({
                type: 'end_interview',
            }));

            this.inputEnabled = false;
            this.pendingEnd = false;
            this.isTyping = true;
            this.typingSource = 'Generating scorecard...';
            this.scrollToBottom();
        },

        resetInterview() {
            if (this.ws) {
                this.ws.close();
                this.ws = null;
            }
            this.screen = 'setup';
            this.messages = [];
            this.summaryData = null;
            this.userInput = '';
            this.inputEnabled = false;
            this.isTyping = false;
            this.questionsAnswered = 0;
            this.totalQuestions = 0;
            this.pendingEnd = false;
            this.connected = false;
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
                const interview = await res.json();
                if (interview.summary_data) {
                    this.jobRole = interview.job_position;
                    this.summaryData = interview.summary_data;
                    this.screen = 'summary';
                    this.messages = (interview.messages || []).map(m => ({
                        type: m.role === 'user' ? 'user' : 'agent',
                        source: m.name || 'interviewer',
                        sourceName: this.capitalize(m.name || 'interviewer'),
                        content: m.content,
                        metadata: {},
                    }));
                }
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
                if (data.error) {
                    this.resumeError = data.error;
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
}
