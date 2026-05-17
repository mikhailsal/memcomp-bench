/**
 * Minimal markdown-to-HTML renderer for conversation text.
 * Handles: bold, italic, inline code, code blocks, links, line breaks.
 */
export function renderMarkdown(text: string): string {
  let html = escapeHtml(text);

  // Code blocks (```)
  html = html.replace(/```(\w*)\n([\s\S]*?)```/g, (_m, _lang, code) =>
    `<pre class="bg-black/30 rounded-lg p-3 overflow-x-auto text-xs font-mono my-2"><code>${code.trim()}</code></pre>`
  );

  // Inline code
  html = html.replace(/`([^`]+)`/g, '<code class="bg-black/20 px-1 py-0.5 rounded text-xs font-mono">$1</code>');

  // Bold
  html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');

  // Italic
  html = html.replace(/\*(.+?)\*/g, '<em>$1</em>');

  // Links
  html = html.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" class="text-sky-400 hover:underline" target="_blank" rel="noopener">$1</a>');

  // Line breaks (double newline = paragraph, single = br)
  html = html
    .split('\n\n')
    .map(p => `<p>${p.replace(/\n/g, '<br />')}</p>`)
    .join('');

  return html;
}

function escapeHtml(text: string): string {
  return text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
}
