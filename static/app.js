function interviewApp() {
    return {
        // State
        screen: 'setup',       // 'setup' | 'chat' | 'summary'
        jobRole: 'AI Engineer',
        messages: [],
        userInput: '',
        inputEnabled: false,
        isTyping: false,
        typingSource: 'Thinking...',
        summaryData: null,
        ws: null,
        questionsAnswered: 0,
        pendingEnd: false,
        connected: false,

        get statusText() {
            if (this.screen === 'summary') return '● Completed';
            if (this.connected) return '● Live';
            return '● Disconnected';
        },
        get statusClass() {
            if (this.screen === 'summary') return 'status-done';
            if (this.connected) return 'status-live';
            return 'status-disconnected';
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
                this.ws.send(JSON.stringify({ job_position: this.jobRole }));
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
            this.pendingEnd = false;
            this.connected = false;
        },

        // Helpers
        scrollToBottom() {
            this.$nextTick(() => {
                const el = this.$refs.messages;
                if (el) el.scrollTop = el.scrollHeight;
            });
        },

        capitalize(s) {
            return s.charAt(0).toUpperCase() + s.slice(1);
        },

        formatCategory(cat) {
            return cat.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
        },

        scoreColor(score) {
            if (score >= 8) return 'score-high';
            if (score >= 5) return 'score-mid';
            return 'score-low';
        },
    };
}
