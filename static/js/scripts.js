// Global variables
let isAnimating = false;
let currentLang = 'en';

// Function to toggle input state
function toggleInputState(disabled) {
    const questionInput = document.getElementById('question-input');
    const langToggleBtn = document.getElementById('lang-toggle');
    
    // Disable/enable input
    questionInput.disabled = disabled;
    questionInput.style.opacity = disabled ? '0.5' : '1';
    questionInput.style.cursor = disabled ? 'not-allowed' : 'text';
    
    // Disable/enable language toggle
    langToggleBtn.disabled = disabled;
    langToggleBtn.style.opacity = disabled ? '0.5' : '1';
    langToggleBtn.style.cursor = disabled ? 'not-allowed' : 'pointer';
}

document.addEventListener('DOMContentLoaded', () => {
    const themeToggleBtn = document.querySelector('#theme-toggle');
    themeToggleBtn?.addEventListener('click', function () {
        applyTheme(!document.documentElement.classList.contains('dark'));
    });
    window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', (e) => applyTheme(e.matches));

    // Initialize language state
    const html = document.documentElement;
    let sources = { en: [], ar: [] };
    let translations = {};

    // Add Enter key handler for question input
    const questionInput = document.getElementById('question-input');
    questionInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            document.getElementById('question-form').dispatchEvent(new Event('submit'));
        }
    });

    // Function to check if chat is empty
    function isChatEmpty() {
        const messagesDiv = document.getElementById('chat-messages');
        // Check if there are any messages other than the welcome message
        const messages = messagesDiv.children;
        if (messages.length === 0) return true;
        if (messages.length === 1 && messages[0].classList.contains('empty-chat-container')) return true;
        return false;
    }

    // Function to update chat layout
    function updateChatLayout() {
        const messagesDiv = document.getElementById('chat-messages');
        const inputArea = document.querySelector('.input-container').parentElement.parentElement;
        
        if (isChatEmpty()) {
            // Clear existing content
            messagesDiv.innerHTML = '';
            
            // Add empty state description
            const emptyContainer = document.createElement('div');
            emptyContainer.className = 'empty-chat-container';
            emptyContainer.innerHTML = `
                <div class="empty-chat-description">
                    <img src="/static/images/logo.png" alt="Assistant" class="welcome-icon">
                    <p class="${currentLang === 'ar' ? 'ar font-arabic' : 'en'}">
                        ${currentLang === 'ar' ? 
                            'مرحباً، أنا مساعد الفتاوى الإسلامية الذكي' :
                            'Hi, I\'m Islamic fatwa AI assistant.'}
                    </p>
                </div>
                <div class="disclaimer-container">
                    <small class="${currentLang === 'ar' ? 'ar font-arabic' : 'en'}">
                        ${currentLang === 'ar' ? 
                            'ملاحظة: قد تحدث بعض الأخطاء في الإجابات. يرجى التحقق دائماً من المصادر المذكورة.' :
                            'Note: Responses may occasionally contain inaccuracies. Always verify with the provided sources.'}
                    </small>
                </div>
            `;
            messagesDiv.appendChild(emptyContainer);
            
            inputArea.classList.add('input-area-centered');
        } else {
            // Remove empty state if there are messages
            const emptyContainer = messagesDiv.querySelector('.empty-chat-container');
            if (emptyContainer) {
                emptyContainer.remove();
            }
            inputArea.classList.remove('input-area-centered');
        }
    }

    // Update layout on page load
    updateChatLayout();

    // Form submission handler
    document.getElementById('question-form').addEventListener('submit', async (e) => {
        e.preventDefault();
        
        if (isAnimating) return; // Prevent submission if animation is in progress
        
        const questionInput = document.getElementById('question-input');
        const question = questionInput.value.trim();
        const sourceSelect = document.getElementById('source-select');
        const source = sourceSelect ? sourceSelect.value : null;
        
        if (!question) return;
        
        isAnimating = true; // Set animation state
        toggleInputState(true); // Disable input
        
        // Remove empty state if this is the first message
        if (isChatEmpty()) {
            const messagesDiv = document.getElementById('chat-messages');
            messagesDiv.innerHTML = '';
            const inputArea = document.querySelector('.input-container').parentElement.parentElement;
            inputArea.classList.remove('input-area-centered');
        }
        
        // Update question header
        const questionHeader = document.getElementById('question-header');
        const headerTitle = questionHeader.querySelector('h2');
        headerTitle.textContent = question;
        headerTitle.setAttribute('dir', detectTextDirection(question));
        questionHeader.style.opacity = '1';
        
        // Add user question to chat
        addMessage('user', question);
        
        // Clear input
        questionInput.value = '';
        
        try {
            // Create and show loading indicator
            const messagesDiv = document.getElementById('chat-messages');
            const loadingDiv = document.createElement('div');
            loadingDiv.className = 'flex justify-center w-full';
            loadingDiv.innerHTML = `
                <div class="inline-flex items-center bg-white dark:bg-gray-800 rounded-full px-3 py-1.5 shadow-sm">
                    <span class="text-gray-700 dark:text-gray-300 ${currentLang === 'en' ? 'en' : 'ar font-arabic'}">${translations.thinking || 'Thinking...'}</span>
                    <div class="flex space-x-1 rtl:space-x-reverse ml-1.5 rtl:ml-0 rtl:mr-1.5">
                        <div class="typing-dot w-1 h-1 bg-blue-600 rounded-full"></div>
                        <div class="typing-dot w-1 h-1 bg-blue-600 rounded-full" style="animation-delay: 0.2s"></div>
                        <div class="typing-dot w-1 h-1 bg-blue-600 rounded-full" style="animation-delay: 0.4s"></div>
                    </div>
                </div>
            `;
            loadingDiv.setAttribute('dir', detectTextDirection(question));
            messagesDiv.appendChild(loadingDiv);
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
            
            // Send question to backend
            const response = await fetch('/ask', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    question: question,
                    provider: source,
                    language: currentLang
                })
            });
            
            const data = await response.json();
            
            // Remove loading message
            loadingDiv.remove();
            
            if (data.error) {
                addMessage('system', data.error);
                isAnimating = false;
                toggleInputState(false);
                return;
            }
            
            // Add response to chat with the full data object
            addMessage('assistant', {
                answer: data.answer,
                sources: data.sources,
                language: data.language
            });
            
        } catch (error) {
            console.error('Error:', error);
            const loadingDiv = document.querySelector('.flex.justify-center');
            if (loadingDiv) {
                loadingDiv.remove();
            }
            addMessage('system', translations.error_internal || 'An error occurred');
            isAnimating = false;
            toggleInputState(false);
        }
    });

    // Fetch translations from backend
    async function fetchTranslations(lang) {
        try {
            const response = await fetch(`/translations/${lang}`);
            translations = await response.json();
            updateTranslations();
        } catch (error) {
            console.error('Error fetching translations:', error);
        }
    }

    // Update UI with current translations
    function updateTranslations() {
        // Update placeholder
        document.getElementById('question-input').placeholder = translations.placeholder;
        
        // Update title
        document.querySelector('h1').innerHTML = `
            <span class="${currentLang === 'en' ? 'en' : 'ar font-arabic'}">${translations.title}</span>
        `;
    }

    // Fetch sources from backend
    async function fetchSources() {
        try {
            const response = await fetch('/sources');
            sources = await response.json();
            updateSourceSelect();
        } catch (error) {
            console.error('Error fetching sources:', error);
        }
    }

    // Update source select options based on current language
    function updateSourceSelect() {
        const sourceSelect = document.getElementById('source-select');
        sourceSelect.innerHTML = '';
        
        const currentSources = sources[currentLang] || [];
        currentSources.forEach(source => {
            const option = document.createElement('option');
            option.value = source.id;
            option.textContent = source.name;
            option.className = currentLang === 'en' ? 'en' : 'ar font-arabic';
            sourceSelect.appendChild(option);
        });
    }

    // Initialize page
    async function initializePage() {
        await Promise.all([
            fetchTranslations(currentLang),
            fetchSources()
        ]);
    }

    // Call initialization on page load
    initializePage();

    // Add language toggle functionality
    document.getElementById('lang-toggle').addEventListener('click', async () => {
        currentLang = currentLang === 'en' ? 'ar' : 'en';
        const dir = currentLang === 'ar' ? 'rtl' : 'ltr';
        
        // Update HTML direction and language
        html.setAttribute('dir', dir);
        html.setAttribute('lang', currentLang);
        
        // Update textarea alignment
        const questionInput = document.getElementById('question-input');
        questionInput.style.textAlign = currentLang === 'ar' ? 'right' : 'left';
        
        // Fetch and update translations
        await fetchTranslations(currentLang);
        
        // Update source select
        updateSourceSelect();

        // Update layout for empty state in new language
        if (isChatEmpty()) {
            updateChatLayout();
        }

        if (currentLang === 'en') {
            Yamli.deyamlify("question-input");
        } else if (currentLang === 'ar') {
            Yamli.yamlify("question-input", { settingsPlacement: "hide" });
        }
    });

    // Add sidebar toggle functionality
    const sidebarToggle = document.getElementById('sidebar-toggle');
    const showSidebarBtn = document.getElementById('show-sidebar');
    const sidebar = document.getElementById('sidebar');
    const mainContent = document.getElementById('main-content');
    let isSidebarVisible = true;

    function toggleSidebar(show) {
        isSidebarVisible = show;
        
        if (!isSidebarVisible) {
            sidebar.classList.add('hidden');
            mainContent.classList.remove('lg:pl-64');
            mainContent.classList.add('lg:pl-0');
            showSidebarBtn.classList.remove('hidden');
        } else {
            sidebar.classList.remove('hidden');
            mainContent.classList.add('lg:pl-64');
            mainContent.classList.remove('lg:pl-0');
            showSidebarBtn.classList.add('hidden');
        }
    }

    sidebarToggle.addEventListener('click', () => toggleSidebar(false));
    showSidebarBtn.addEventListener('click', () => toggleSidebar(true));
});

function applyTheme(darkMode) {
    if (darkMode) {
        document.documentElement.classList.remove('light');
        document.documentElement.classList.add('dark');
    } else {
        document.documentElement.classList.remove('dark');
        document.documentElement.classList.add('light');
    }
}

// Start with light mode
applyTheme(false);

// Configure marked.js to open links in new tab
marked.setOptions({
    breaks: true,
    gfm: true
});

function addMessage(type, content) {
    const messagesDiv = document.getElementById('chat-messages');
    const messageDiv = document.createElement('div');
    
    let text, sources, direction;
    if (type === 'assistant' && typeof content === 'object') {
        text = content.answer;
        sources = content.sources;
        direction = content.language === 'ar' ? 'rtl' : 'ltr';
    } else {
        text = content;
        direction = detectTextDirection(text);
    }
    
    const isArabic = direction === 'rtl';
    
    messageDiv.classList.add(
        'message', 
        `${type}-message`,
        'p-4',
        'rounded-lg',
        type === 'user' ? 'shadow-[0_2px_10px_rgba(0,0,0,0.1)]' : 'shadow-sm'
    );
    
    if (type === 'assistant') {
        messageDiv.classList.add('bg-white', 'dark:bg-gray-800', 'text-gray-900', 'dark:text-gray-100');
        
        // Create assistant message container with icon
        const containerDiv = document.createElement('div');
        containerDiv.className = 'assistant-message-container';
        
        // Add assistant icon
        const iconImg = document.createElement('img');
        iconImg.src = '/static/images/logo.png';
        iconImg.className = 'assistant-icon';
        iconImg.alt = 'Assistant';
        containerDiv.appendChild(iconImg);
        
        // Create content container
        const contentDiv = document.createElement('div');
        contentDiv.className = 'flex-1';
        
        // Create answer div with markdown
        const answerDiv = document.createElement('div');
        answerDiv.classList.add('answer-section');
        answerDiv.style.opacity = '1';
        
        // Create sources div if there are sources
        const sourcesDiv = sources && sources.length > 0 ? document.createElement('div') : null;
        if (sourcesDiv) {
            sourcesDiv.classList.add('sources-section');
            sourcesDiv.style.opacity = '1';
            sourcesDiv.style.display = 'none'; // Hide initially
        }
        
        // Add divs to content container
        contentDiv.appendChild(answerDiv);
        if (sourcesDiv) {
            contentDiv.appendChild(sourcesDiv);
        }
        
        // Add content container to message container
        containerDiv.appendChild(contentDiv);
        
        // Add container to message div
        messageDiv.appendChild(containerDiv);

        // Set direction and text formatting
        messageDiv.setAttribute('dir', direction);
        
        if (isArabic) {
            messageDiv.style.fontFamily = "'Noto Naskh Arabic', serif";
            messageDiv.style.fontSize = '1.1rem';
            messageDiv.style.lineHeight = '1.8';
            text = text.replace(/^الجواب: /, '');
        } else {
            text = text.replace(/^Answer: /, '');
        }

        // Prepare the content
        const parsedAnswer = marked.parse(text);
        const sourcesTitle = isArabic ? 'المصادر' : 'Sources';
        const sourcesContent = sources && sources.length > 0 ? 
            `<p class="text-gray-700 dark:text-gray-300">${sourcesTitle}</p><ul>` + 
            sources.map(source => 
                `<li><a href="${source.url}" target="_blank" rel="noopener noreferrer" class="text-blue-600 dark:text-blue-400 hover:underline">${source.title}</a></li>`
            ).join('') + 
            '</ul>' : '';

        // Typewriter effect function
        function typeWriter(element, content, onComplete) {
            let tempDiv = document.createElement('div');
            tempDiv.innerHTML = content;
            
            // Split content into HTML tags and text
            const tokens = [];
            let currentToken = '';
            let inTag = false;
            let listType = null;
            let listCounter = 0;
            let currentListItem = '';
            let isArabic = element.closest('[dir="rtl"]') !== null;
            
            // Function to convert to Arabic numbers if needed
            function formatNumber(num) {
                if (!isArabic) return num + '. ';
                const arabicNumbers = {
                    '0': '٠', '1': '١', '2': '٢', '3': '٣', '4': '٤',
                    '5': '٥', '6': '٦', '7': '٧', '8': '٨', '9': '٩'
                };
                return num.toString().replace(/[0-9]/g, d => arabicNumbers[d]) + ' . ';
            }
            
            for (let i = 0; i < content.length; i++) {
                const char = content[i];
                if (char === '<') {
                    if (currentToken) tokens.push({ type: 'text', content: currentToken });
                    currentToken = char;
                    inTag = true;
                    
                    // Look ahead to check tag type
                    const nextContent = content.substring(i, i + 10);
                    if (nextContent.startsWith('<ol')) {
                        listType = 'ordered';
                        listCounter = 0;
                    } else if (nextContent.startsWith('<ul')) {
                        listType = 'unordered';
                    } else if (nextContent.startsWith('</ol>') || nextContent.startsWith('</ul>')) {
                        listType = null;
                        listCounter = 0;
                    } else if (nextContent.startsWith('<li>')) {
                        if (listType === 'ordered') {
                            listCounter++;
                            currentListItem = formatNumber(listCounter);
                        } else if (listType === 'unordered' && !element.classList.contains('sources-section')) {
                            currentListItem = '• ';
                        }
                    }
                } else if (char === '>' && inTag) {
                    currentToken += char;
                    tokens.push({ type: 'tag', content: currentToken });
                    
                    // Add list marker after the <li> tag, ensuring no extra space
                    if (currentToken === '<li>' && currentListItem) {
                        tokens.push({ type: 'text', content: currentListItem });
                        currentListItem = '';
                    }
                    
                    currentToken = '';
                    inTag = false;
                } else {
                    currentToken += char;
                    if (!inTag && (i === content.length - 1 || content[i + 1] === '<')) {
                        // Trim any leading spaces in list items to prevent extra spacing
                        if (listType) {
                            currentToken = currentToken.trimStart();
                        }
                        tokens.push({ type: 'text', content: currentToken });
                        currentToken = '';
                    }
                }
            }
            
            let currentIndex = 0;
            let currentTextIndex = 0;
            let currentText = '';
            
            function type() {
                if (currentIndex >= tokens.length) {
                    if (onComplete) onComplete();
                    return;
                }

                const token = tokens[currentIndex];
                
                if (token.type === 'tag') {
                    // Add full tag immediately
                    currentText += token.content;
                    currentIndex++;
                    element.innerHTML = currentText;
                    requestAnimationFrame(type);
                } else {
                    // Type text character by character
                    if (currentTextIndex < token.content.length) {
                        currentText += token.content[currentTextIndex];
                        currentTextIndex++;
                        element.innerHTML = currentText;
                        setTimeout(type, 15);
                    } else {
                        currentTextIndex = 0;
                        currentIndex++;
                        requestAnimationFrame(type);
                    }
                }
            }
            
            // Start typing with error handling
            try {
                type();
            } catch (error) {
                console.error('Typewriter error:', error);
                // Fallback: show all content immediately
                element.innerHTML = content;
                if (onComplete) onComplete();
            }
        }

        // Start typewriter for answer
        typeWriter(answerDiv, parsedAnswer, () => {
            // After answer is complete, show sources if they exist
            if (sourcesDiv && sourcesContent) {
                sourcesDiv.style.display = 'block'; // Show sources section after answer is complete
                typeWriter(sourcesDiv, sourcesContent, () => {
                    // Re-enable everything after all typing is complete
                    isAnimating = false; // Reset animation state
                    toggleInputState(false); // Re-enable input and language toggle
                });
            } else {
                // Re-enable everything if no sources
                isAnimating = false; // Reset animation state
                toggleInputState(false); // Re-enable input and language toggle
            }
            
            // Ensure the message div is visible and scroll into view
            messageDiv.style.opacity = '1';
            messageDiv.scrollIntoView({ behavior: 'smooth', block: 'end' });
        });
    } else if (type === 'user') {
        messageDiv.classList.add('bg-blue-100', 'dark:bg-blue-900', 'text-gray-900', 'dark:text-gray-100', 'text-lg');
        messageDiv.setAttribute('dir', direction);
        
        if (isArabic) {
            messageDiv.style.fontFamily = "'Noto Naskh Arabic', serif";
            // messageDiv.style.fontSize = '1.1rem';
            // messageDiv.style.lineHeight = '1.8';
        }
        
        messageDiv.textContent = text;
        // Only apply RTL positioning for new messages in Arabic mode
        if (currentLang === 'ar') {
            messageDiv.style.marginRight = 'auto';
            messageDiv.style.marginLeft = '20%';
        } else {
            messageDiv.style.marginLeft = 'auto';
            messageDiv.style.marginRight = '20%';
        }
    }
    
    messagesDiv.appendChild(messageDiv);
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
}

function detectTextDirection(text) {
    // First character range for Arabic
    const arabic = /[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]/;
    // Check if the first non-whitespace character is Arabic
    const firstChar = text.trim().charAt(0);
    return arabic.test(firstChar) ? 'rtl' : 'ltr';
}