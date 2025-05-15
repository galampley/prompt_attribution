"""HTML heat-map visualization for prompt attribution."""

import html
import json
from typing import Dict, List, Optional, Tuple
import uuid
import re
from textwrap import shorten
from itertools import chain

from ..engine import Run
from ..segmenter import Span
from ..scorer import Scorer


class HeatmapVisualizer:
    """Renders HTML heat-maps of prompt attribution results.
    
    Features:
    - Inline CSS for displaying segments with impact-based coloring
    - Optional tooltips with score details
    - Configurable color schemes
    """
    
    def __init__(
        self,
        color_scheme: str = "red",
        include_tooltips: bool = True,
        min_opacity: float = 0.05,
    ):
        """Initialize the visualizer.
        
        Args:
            color_scheme: Color scheme for heat map ('red', 'blue', or 'gradient')
            include_tooltips: Whether to include hover tooltips with details
            min_opacity: Minimum opacity for the lowest-impact segments
        """
        self.color_scheme = color_scheme
        self.include_tooltips = include_tooltips
        self.min_opacity = min_opacity
        
        # Define color schemes
        self.color_schemes = {
            "red": {
                "base_color": "rgba(255, 0, 0, {opacity})",
                "text_color": "#000000",
            },
            "blue": {
                "base_color": "rgba(0, 0, 255, {opacity})",
                "text_color": "#000000",
            },
            "gradient": {
                "high": "rgba(255, 0, 0, {opacity})",  # Red for high impact
                "medium": "rgba(255, 165, 0, {opacity})",  # Orange for medium
                "low": "rgba(0, 0, 255, {opacity})",  # Blue for low impact
                "text_color": "#000000",
            },
        }
    
    def visualize_run(self, run: Run) -> str:
        """Generate an HTML heat-map for a run.
        
        Args:
            run: The run with attribution results
            
        Returns:
            HTML string with heat-map visualization
        """
        # Get results and segments
        results = run.ablation_results
        segments = run.segments
        
        # Use the scorer to normalize scores
        scorer = Scorer(run.completion)
        normalized_results = scorer.get_normalized_scores(results)
        
        # Create a mapping from span_id to normalized score
        score_map = {r["span_id"]: r["normalized_score"] for r in normalized_results}
        
        # Also create a map for the raw delta cosine values
        delta_cos_map = {r["span_id"]: r["delta_cos"] for r in results}
        
        # Sort segments by their starting position
        sorted_segments = sorted(segments, key=lambda s: s["start"])
        
        # Generate HTML sections
        model_response_html = self._generate_model_response_section(run.completion, run.response_control, score_map)
        heat_map_html = self._generate_html(run.prompt, sorted_segments, score_map)
        matrix_html = self._generate_influence_matrix(run, segments)
        sentence_table_html = self._generate_sentence_table(
            run.completion, run.response_control, run.response_sentence_deltas, segments, score_map, run.rewrite_suggestions
        )
        table_html = self._generate_table(sorted_segments, score_map, delta_cos_map)
        
        # Combine all sections in logical order:
        # 1. First the heat map of the prompt
        # 2. Then the model's response
        # 3. Then the influence matrix
        # 4. Then the sentence table
        # 5. Finally the detailed score table
        return (
            heat_map_html
            + "\n"
            + model_response_html
            + "\n"
            + matrix_html
            + "\n"
            + sentence_table_html
            + "\n"
            + table_html
        )
    
    def _generate_html(
        self, 
        prompt: str, 
        segments: List[Dict], 
        score_map: Dict[int, float]
    ) -> str:
        """Generate HTML for heat-map visualization.
        
        Args:
            prompt: The original prompt text
            segments: List of prompt segments
            score_map: Mapping of span_id to impact score
            
        Returns:
            HTML string with heat-map visualization
        """
        # Start building HTML
        unique_id = str(uuid.uuid4())[:8]
        html_parts = [
            f'<div id="prompt-heatmap-{unique_id}" class="prompt-container">',
            '<h3>Prompt</h3>',
            '<div class="prompt-content">',
            '<style>',
            '.prompt-container { margin: 1.5em 0; padding: 1em; border: 1px solid #ddd; border-radius: 4px; }',
            '.prompt-container h3 { margin-top: 0; display: flex; justify-content: space-between; align-items: center; }',
            '.prompt-content { font-family: monospace; white-space: pre-wrap; line-height: 1.5; padding: 1em; border: 1px solid #ccc; border-radius: 4px; }',
            '.segment { padding: 2px 0; display: inline; }',
            '.segment-tooltip { position: absolute; background: #f8f8f8; border: 1px solid #ddd; padding: 8px; border-radius: 4px; font-size: 12px; max-width: 300px; z-index: 100; box-shadow: 2px 2px 5px rgba(0,0,0,0.2); }',
            '</style>',
        ]
        
        # Add segments with appropriate coloring
        last_end = 0
        
        for segment in segments:
            span_id = segment["id"]
            start = segment["start"]
            end = segment["end"]
            text = segment["text"]
            
            # Handle any text between segments
            if start > last_end:
                between_text = prompt[last_end:start]
                html_parts.append(html.escape(between_text))
            
            # Calculate opacity based on score
            score = score_map.get(span_id, 0.0)
            opacity = self.min_opacity + score * (1.0 - self.min_opacity)
            
            # Get color based on scheme
            bg_color = self._get_color_for_score(score, opacity)
            
            # Create tooltip if enabled
            tooltip_attr = ""
            if self.include_tooltips:
                tooltip = f"Segment {span_id}: Impact score {score:.4f}"
                tooltip_attr = f' title="{tooltip}" data-score="{score:.4f}"'
            
            # Add the segment with styling
            segment_html = (
                f'<span class="segment" style="background-color: {bg_color};"{tooltip_attr}>'
                f'{html.escape(text)}</span>'
            )
            html_parts.append(segment_html)
            
            last_end = end
        
        # Add any remaining text
        if last_end < len(prompt):
            html_parts.append(html.escape(prompt[last_end:]))
        
        # Close tags - fixing the structure to match other sections
        html_parts.extend(['</div>', '</div>'])
        
        return '\n'.join(html_parts)
    
    def _get_color_for_score(self, score: float, opacity: float) -> str:
        """Get the appropriate color for a score.
        
        Args:
            score: Normalized impact score (0-1)
            opacity: Calculated opacity value
            
        Returns:
            CSS color string
        """
        scheme = self.color_schemes.get(self.color_scheme, self.color_schemes["red"])
        
        if self.color_scheme == "gradient":
            if score > 0.66:
                return scheme["high"].format(opacity=opacity)
            elif score > 0.33:
                return scheme["medium"].format(opacity=opacity)
            else:
                return scheme["low"].format(opacity=opacity)
        else:
            return scheme["base_color"].format(opacity=opacity)
    
    def _generate_table(
        self,
        segments: List[Dict],
        score_map: Dict[int, float],
        delta_cos_map: Dict[int, float]
    ) -> str:
        """Generate a sortable table of segment scores.
        
        Args:
            segments: List of prompt segments
            score_map: Mapping of span_id to normalized score
            delta_cos_map: Mapping of span_id to raw delta cosine values
            
        Returns:
            HTML string with table visualization
        """
        # Start building HTML for the table
        table_id = f"segment-table-{str(uuid.uuid4())[:8]}"
        func = f"sort_{table_id.replace('-', '_')}"
        html_parts = [
            '<div class="segment-table-container">',
            '<h3>Prompt Segment Impact Scores</h3>',
            f'<table id="{table_id}" class="segment-table">',
            '<thead>',
            '<tr>',
            f'<th onclick="{func}(0, \'numeric\')" class="sortable">ID</th>',
            f'<th onclick="{func}(1, \'numeric\')" class="sortable">Raw Impact</th>',
            f'<th onclick="{func}(2, \'numeric\')" class="sortable">Normalized Score</th>',
            f'<th onclick="{func}(3, \'text\')" class="sortable">Preview</th>',
            '</tr>',
            '</thead>',
            '<tbody>'
        ]
        
        # Add rows for each segment
        for segment in segments:
            span_id = segment["id"]
            raw_score = delta_cos_map.get(span_id, 0.0)
            norm_score = score_map.get(span_id, 0.0)
            
            # Get a preview of the segment text (first 50 chars)
            text = segment["text"]
            preview = text[:50].replace("\n", " ")
            if len(text) > 50:
                preview += "..."
            
            # Add table row
            row = (
                f'<tr data-id="{span_id}">',
                f'<td>{span_id}</td>',
                f'<td>{raw_score:.4f}</td>',
                f'<td>{norm_score:.4f}</td>',
                f'<td>{html.escape(preview)}</td>',
                '</tr>'
            )
            html_parts.extend(row)
        
        # Close the table
        html_parts.extend(['</tbody>', '</table>'])
        
        # Add sorting JavaScript
        js = f"""
        <script>
        function {func}(columnIndex, type) {{
            const table = document.getElementById('{table_id}');
            const tbody = table.querySelector('tbody');
            const rows = Array.from(tbody.querySelectorAll('tr'));
            
            // Get the current sort direction
            const th = table.querySelectorAll('th')[columnIndex];
            const currentDir = th.getAttribute('data-sort') || 'asc';
            const newDir = currentDir === 'asc' ? 'desc' : 'asc';
            
            // Update all headers to remove sort indicators
            table.querySelectorAll('th').forEach(header => {{
                header.removeAttribute('data-sort');
                header.classList.remove('sorted-asc', 'sorted-desc');
            }});
            
            // Set the new sort direction
            th.setAttribute('data-sort', newDir);
            th.classList.add(newDir === 'asc' ? 'sorted-asc' : 'sorted-desc');
            
            // Sort the rows
            rows.sort((a, b) => {{
                let aValue = a.cells[columnIndex].textContent;
                let bValue = b.cells[columnIndex].textContent;
                
                if (type === 'numeric') {{
                    aValue = parseFloat(aValue);
                    bValue = parseFloat(bValue);
                }}
                
                if (aValue < bValue) return newDir === 'asc' ? -1 : 1;
                if (aValue > bValue) return newDir === 'asc' ? 1 : -1;
                return 0;
            }});
            
            // Re-append rows in the new order
            rows.forEach(row => tbody.appendChild(row));
        }}
        
        // Sort by raw impact (descending) on load
        window.addEventListener('load', function() {{
            const table = document.getElementById('{table_id}');
            const rawImpactHeader = table.querySelectorAll('th')[1];
            rawImpactHeader.click();  // Initial sort by raw impact
            rawImpactHeader.click();  // Click again to make it descending
        }});
        </script>
        """
        
        # Add table CSS
        css = """
        <style>
        .segment-table-container {{
            margin-top: 2em;
            padding: 1em;
            border: 1px solid #ddd;
            border-radius: 4px;
        }}
        .segment-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 0.5em;
            font-size: 14px;
        }}
        .segment-table th, .segment-table td {{
            padding: 8px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        .segment-table th {{
            background-color: #f5f5f5;
            cursor: pointer;
            position: relative;
        }}
        .segment-table th.sortable:hover {{
            background-color: #e5e5e5;
        }}
        .segment-table th.sorted-asc::after {{
            content: " ↑";
            font-size: 0.8em;
        }}
        .segment-table th.sorted-desc::after {{
            content: " ↓";
            font-size: 0.8em;
        }}
        .segment-table tr:hover {{
            background-color: #f9f9f9;
        }}
        </style>
        """
        
        html_parts.extend([css, js, '</div>'])
        return '\n'.join(html_parts)
    
    def _generate_model_response_section(self, completion: str, response_control: List[int], score_map: Dict[int, float]) -> str:
        """Generate HTML section showing the model's response.
        
        Args:
            completion: The model's response text
            response_control: List of segment IDs that control the response
            score_map: Mapping of span_id to normalized score
            
        Returns:
            HTML string with the model response section
        """
        # Create a unique ID for the collapsible container
        container_id = f"model-response-{str(uuid.uuid4())[:8]}"
        collapse_id = f"collapse-{container_id}"
        
        # Split completion into sentences
        sentences = re.split(r'(?<=[.!?])\s+', completion.strip())
        colored_sentences = []
        for idx, sentence in enumerate(sentences):
            seg_id = response_control[idx] if idx < len(response_control) else -1
            if seg_id != -1:
                score = score_map.get(seg_id, 0.0)
                opacity = self.min_opacity + score * (1.0 - self.min_opacity)
                bg_color = self._get_color_for_score(score, opacity)
                colored_sentence = f'<span style="background-color: {bg_color};">{html.escape(sentence)}</span>'
            else:
                colored_sentence = html.escape(sentence)
            colored_sentences.append(colored_sentence)
        colored_completion = ' '.join(colored_sentences)
        
        html_str = f"""
        <div class="model-response-container" id="{container_id}">
            <h3>
                Model Response
                <button class="toggle-button" onclick="toggleCollapse('{collapse_id}')">
                    <span class="toggle-text">Hide</span>
                </button>
            </h3>
            <div class="model-response-content" id="{collapse_id}">
                <pre class="model-response">{colored_completion}</pre>
            </div>
        </div>
        
        <style>
        .model-response-container {{
            margin: 1.5em 0;
            padding: 1em;
            border: 1px solid #ddd;
            border-radius: 4px;
        }}
        .model-response-container h3 {{
            margin-top: 0;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .model-response {{
            white-space: pre-wrap;
            background-color: #f9f9f9;
            padding: 1em;
            border-radius: 4px;
            margin: 0;
            font-size: 14px;
            overflow-x: auto;
        }}
        .toggle-button {{
            background-color: #f5f5f5;
            border: 1px solid #ddd;
            padding: 0.3em 0.6em;
            border-radius: 4px;
            cursor: pointer;
            font-size: 12px;
        }}
        .toggle-button:hover {{
            background-color: #e5e5e5;
        }}
        </style>
        
        <script>
        function toggleCollapse(id) {{
            const element = document.getElementById(id);
            const button = element.previousElementSibling.querySelector('.toggle-button');
            const textSpan = button.querySelector('.toggle-text');
            
            if (element.style.display === 'none') {{
                element.style.display = 'block';
                textSpan.textContent = 'Hide';
            }} else {{
                element.style.display = 'none';
                textSpan.textContent = 'Show';
            }}
        }}
        </script>
        """
        
        return html_str
    
    def _generate_sentence_table(
        self,
        completion: str,
        control: List[int],
        sentence_deltas: List[float],
        segments: List[Dict],
        score_map: Dict[int, float],
        rewrite_suggestions: Optional[Dict[str, List[str]]] = None
    ) -> str:
        """Generate a sortable table mapping response sentences to prompt segments.
        
        Args:
            completion: The model's response text
            control: List of segment IDs that control the response
            sentence_deltas: List of sentence-level deltas
            segments: List of prompt segments
            score_map: Mapping of span_id to normalized score
            rewrite_suggestions: Optional dict of rewrite suggestions
            
        Returns:
            HTML string with sentence table visualization
        """
        # Start building HTML for the sentence table
        table_id = f"sentence-table-{str(uuid.uuid4())[:8]}"
        modal_id = f"modal-{str(uuid.uuid4())[:8]}"
        func = f"sort_{table_id.replace('-', '_')}"
        html_parts = [
            '<div class="sentence-table-container">',
            '<h3>Response Sentence Impact Scores</h3>',
            f'<table id="{table_id}" class="sentence-table">',
            '<thead>',
            '<tr>',
            f'<th onclick="{func}(0, \'numeric\')" class="sortable">Response&nbsp;Sentence&nbsp;#</th>',
            f'<th onclick="{func}(1, \'text\')" class="sortable">Sentence&nbsp;Preview</th>',
            f'<th onclick="{func}(2, \'numeric\')" class="sortable">Prompt&nbsp;Segment&nbsp;ID</th>',
            f'<th onclick="{func}(3, \'text\')" class="sortable">Prompt&nbsp;Preview</th>',
            f'<th onclick="{func}(4, \'numeric\')" class="sortable">Sentence&nbsp;Δ</th>',
            f'<th onclick="{func}(5, \'numeric\')" class="sortable">Seg&nbsp;Norm&nbsp;Score</th>',
            f'<th class="no-sort">Actions</th>',
            '</tr>',
            '</thead>',
            '<tbody>'
        ]
        
        # Add rows for each sentence
        sentences = re.split(r'(?<=[.!?])\s+', completion.strip())
        for idx, sentence_delta in enumerate(sentence_deltas):
            sent_text = sentences[idx] if idx < len(sentences) else ""
            sent_preview = shorten(sent_text, 80, placeholder="…")
            seg_id = control[idx] if idx < len(control) else -1
            norm_score = score_map.get(seg_id, 0.0)
            
            # find prompt preview
            prompt_preview = ""
            if seg_id != -1:
                for seg in segments:
                    if seg["id"] == seg_id:
                        prompt_preview = shorten(seg["text"].replace("\n", " "), 80, placeholder="…")
                        break
            
            # Look for existing rewrite suggestions
            has_rewrites = False
            if rewrite_suggestions:
                key = f"seg{seg_id}_sent{idx}"
                has_rewrites = key in rewrite_suggestions and rewrite_suggestions[key]
            
            # Add rewrite button - only if we have a valid segment ID
            action_cell = ""
            if seg_id >= 0:
                btn_class = "rewrite-btn has-rewrites" if has_rewrites else "rewrite-btn"
                action_cell = f'<button class="{btn_class}" data-sentence-idx="{idx}" data-span-id="{seg_id}" onclick="showRewriteModal(this, {idx}, {seg_id})">💡</button>'
            
            # Add table row
            row = (
                f'<tr data-id="{idx}">',
                f'<td>{idx}</td>',
                f'<td>{html.escape(sent_preview)}</td>',
                f'<td>{seg_id}</td>',
                f'<td>{html.escape(prompt_preview)}</td>',
                f'<td>{sentence_delta:.4f}</td>',
                f'<td>{norm_score:.4f}</td>',
                f'<td class="action-cell">{action_cell}</td>',
                '</tr>'
            )
            html_parts.extend(row)
        
        # Close the table
        html_parts.extend(['</tbody>', '</table>'])
        
        # Add sorting JavaScript
        js = f"""
        <script>
        function {func}(columnIndex, type) {{
            const table = document.getElementById('{table_id}');
            const tbody = table.querySelector('tbody');
            const rows = Array.from(tbody.querySelectorAll('tr'));
            
            // Get the current sort direction
            const th = table.querySelectorAll('th')[columnIndex];
            const currentDir = th.getAttribute('data-sort') || 'asc';
            const newDir = currentDir === 'asc' ? 'desc' : 'asc';
            
            // Update all headers to remove sort indicators
            table.querySelectorAll('th').forEach(header => {{
                header.removeAttribute('data-sort');
                header.classList.remove('sorted-asc', 'sorted-desc');
            }});
            
            // Set the new sort direction
            th.setAttribute('data-sort', newDir);
            th.classList.add(newDir === 'asc' ? 'sorted-asc' : 'sorted-desc');
            
            // Sort the rows
            rows.sort((a, b) => {{
                let aValue = a.cells[columnIndex].textContent;
                let bValue = b.cells[columnIndex].textContent;
                
                if (type === 'numeric') {{
                    aValue = parseFloat(aValue);
                    bValue = parseFloat(bValue);
                }}
                
                if (aValue < bValue) return newDir === 'asc' ? -1 : 1;
                if (aValue > bValue) return newDir === 'asc' ? 1 : -1;
                return 0;
            }});
            
            // Re-append rows in the new order
            rows.forEach(row => tbody.appendChild(row));
        }}
        
        // Sort by sentence delta (descending) on load
        window.addEventListener('load', function() {{
            const table = document.getElementById('{table_id}');
            const deltaHeader = table.querySelectorAll('th')[4];
            deltaHeader.click();
            deltaHeader.click();
        }});
        
        // Rewrite modal functionality
        function showRewriteModal(btn, sentIdx, spanId) {{
            // TODO: In a real implementation, this would fetch suggestions from the server
            // or display a modal to enter a comment and request suggestions
            alert(`Request rewrite for sentence ${{sentIdx}} influenced by span ${{spanId}}\\n\\nIn a full implementation, this would show a modal dialog to enter what's wrong and fetch suggestions.`);
        }}
        </script>
        """
        
        # Add rewrite modal HTML
        modal_html = f"""
        <div id="{modal_id}" class="rewrite-modal">
            <div class="modal-content">
                <div class="modal-header">
                    <h3>Request Rewrite Suggestion</h3>
                    <span class="close-modal">&times;</span>
                </div>
                <div class="modal-body">
                    <div class="section">
                        <h4>Response Sentence</h4>
                        <div id="sentence-preview" class="preview-box"></div>
                    </div>
                    <div class="section">
                        <h4>Influential Prompt Segment</h4>
                        <div id="span-preview" class="preview-box"></div>
                    </div>
                    <div class="section">
                        <h4>What's wrong with this response?</h4>
                        <textarea id="user-comment" placeholder="Explain what's wrong with this sentence..."></textarea>
                    </div>
                    <div class="actions">
                        <button id="get-suggestions-btn">Get Rewrite Suggestions</button>
                    </div>
                    <div id="suggestions-container" class="section" style="display:none;">
                        <h4>Suggested Rewrites</h4>
                        <div id="suggestions-list" class="suggestions-list"></div>
                    </div>
                </div>
            </div>
        </div>
        """
        
        # Add table CSS
        css = """
        <style>
        .sentence-table-container {
            margin-top: 2em;
            padding: 1em;
            border: 1px solid #ddd;
            border-radius: 4px;
        }
        .sentence-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 0.5em;
            font-size: 14px;
        }
        .sentence-table th, .sentence-table td {
            padding: 8px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        .sentence-table th {
            background-color: #f5f5f5;
            cursor: pointer;
            position: relative;
        }
        .sentence-table th.no-sort {
            cursor: default;
        }
        .sentence-table th.sortable:hover {
            background-color: #e5e5e5;
        }
        .sentence-table th.sorted-asc::after {
            content: " ↑";
            font-size: 0.8em;
        }
        .sentence-table th.sorted-desc::after {
            content: " ↓";
            font-size: 0.8em;
        }
        .sentence-table tr:hover {
            background-color: #f9f9f9;
        }
        .action-cell {
            text-align: center;
        }
        .rewrite-btn {
            background: #f0f0f0;
            border: 1px solid #ccc;
            border-radius: 4px;
            padding: 4px 8px;
            cursor: pointer;
            font-size: 14px;
        }
        .rewrite-btn:hover {
            background: #e0e0e0;
        }
        .rewrite-btn.has-rewrites {
            background: #e6f7ff;
            border-color: #91d5ff;
        }
        
        /* Modal styles */
        .rewrite-modal {
            display: none;
            position: fixed;
            z-index: 1000;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0,0,0,0.4);
        }
        .modal-content {
            background-color: #fefefe;
            margin: 10% auto;
            padding: 20px;
            border: 1px solid #888;
            border-radius: 5px;
            width: 80%;
            max-width: 700px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        .modal-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 1px solid #eee;
        }
        .modal-header h3 {
            margin: 0;
        }
        .close-modal {
            font-size: 24px;
            cursor: pointer;
        }
        .section {
            margin-bottom: 20px;
        }
        .preview-box {
            padding: 10px;
            background: #f9f9f9;
            border: 1px solid #ddd;
            border-radius: 4px;
            min-height: 50px;
            white-space: pre-wrap;
        }
        #user-comment {
            width: 100%;
            min-height: 80px;
            padding: 8px;
            margin-top: 5px;
            border: 1px solid #ddd;
            border-radius: 4px;
        }
        .actions {
            display: flex;
            justify-content: flex-end;
            margin-top: 15px;
        }
        #get-suggestions-btn {
            padding: 8px 16px;
            background-color: #1a73e8;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
        }
        #get-suggestions-btn:hover {
            background-color: #1558b3;
        }
        .suggestions-list {
            display: flex;
            flex-direction: column;
            gap: 10px;
        }
        .suggestion {
            padding: 10px;
            background: #f0f8ff;
            border: 1px solid #b3d9ff;
            border-radius: 4px;
            position: relative;
        }
        .copy-btn {
            position: absolute;
            top: 5px;
            right: 5px;
            background: transparent;
            border: none;
            cursor: pointer;
            color: #666;
        }
        </style>
        """
        
        html_parts.extend([css, js, modal_html, '</div>'])
        return '\n'.join(html_parts)
    
    def _generate_influence_matrix(self, run: Run, segments: List[Dict]) -> str:
        """Render a tiny heat-map grid of segment→sentence influence."""
        # Build matrix from ablation results
        if not run.ablation_results or not run.response_sentence_deltas:
            return ""
        # Determine sentence count (columns)
        num_sent = len(run.response_sentence_deltas)
        num_seg = len(segments)
        # init matrix with zeros
        matrix = [[0.0 for _ in range(num_sent)] for _ in range(num_seg)]
        for res in run.ablation_results:
            sid = res["span_id"]
            deltas = res.get("sentence_deltas", [])
            for j, d in enumerate(deltas):
                if j < num_sent and sid < num_seg:
                    matrix[sid][j] = d
        # normalise to 0-1 global
        max_delta = max(chain.from_iterable(matrix)) or 1.0
        norm = [[cell / max_delta for cell in row] for row in matrix]
        # Build HTML
        container_id = f"matrix-{str(uuid.uuid4())[:8]}"
        cell_px = 56
        html_parts = [
            f'<div class="matrix-container" id="{container_id}">',
            '<h3>Prompt Segment ↔ Response Sentence Influence Matrix</h3>',
            '<div class="matrix-scroll">',
            '<table class="influence-matrix">'
        ]
        # header row
        header_cells = ["<th class='corner'></th>"]
        for j in range(num_sent):
            header_cells.append(f"<th class='col-header'>{j}</th>")
        html_parts.append("<tr>" + "".join(header_cells) + "</tr>")

        # data rows with row header
        for i, row in enumerate(norm):
            html_parts.append(f"<tr><th class='row-header'>{i}</th>")
            for cell in row:
                opacity = self.min_opacity + cell * (1.0 - self.min_opacity)
                # Use grayscale instead of color scheme
                gray_color = f"rgba(0, 0, 0, {opacity})"
                html_parts.append(
                    f'<td class="cell" style="background-color:{gray_color}" title="Δ={cell*max_delta:.4f}"></td>'
                )
            html_parts.append("</tr>")
        html_parts.extend([
            '</table></div>',
            '<style>',
            '.matrix-container { margin-top:2em; }',
            '.matrix-scroll { overflow-x:auto; border:1px solid #ddd; padding:4px; }',
            '.influence-matrix { border-collapse: collapse; }',
            f'.cell {{ width:{cell_px}px; height:{cell_px}px; }}',
            '.influence-matrix th { font-size: 12px; position: sticky; z-index: 2; background:#f5f5f5; }',
            '.col-header { top:0; }',
            '.row-header { left:0; }',
            '.corner { top:0; left:0; }',
            '</style>',
            '</div>'
        ])
        return '\n'.join(html_parts)
    
    def save_html(self, run: Run, output_path: str) -> None:
        """Save heat-map visualization to an HTML file.
        
        Args:
            run: The run with attribution results
            output_path: Path to save the HTML file
        """
        html_content = self.visualize_run(run)
        
        # Wrap in a complete HTML document
        full_html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Prompt Attribution Heat Map</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif; margin: 0; padding: 20px; }}
        h1 {{ font-size: 1.5em; margin-bottom: 1em; }}
        h3 {{ font-size: 1.2em; margin-bottom: 0.5em; }}
    </style>
</head>
<body>
    <h1>Prompt Attribution Heat Map</h1>
    {html_content}
    <div style="margin-top: 20px; font-size: 0.8em; color: #666;">
        <p>Color intensity indicates segment impact on model output.</p>
    </div>
</body>
</html>"""
        
        with open(output_path, "w") as f:
            f.write(full_html) 