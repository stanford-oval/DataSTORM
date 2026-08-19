import dspy
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Set, Union
import json
import os
import csv
import io

from .collaborative_storm_utils import clean_up_section
from ...dataclass import KnowledgeBase, KnowledgeNode
import requests


def _csv_text_to_markdown_table(
    csv_text: str,
    *,
    max_rows: int = 50,
    max_cols: int = 25,
    max_cell_chars: int = 200,
) -> str:
    """
    Best-effort conversion of CSV-like text to a Markdown table.

    If parsing fails or the content doesn't look tabular, returns the original text.
    """
    raw = (csv_text or "").strip()
    if not raw:
        return raw

    lowered = raw[:1000].lower()
    if "<table" in lowered or "<tr" in lowered:
        return raw

    # Avoid double-converting if it's already a Markdown table.
    if raw.lstrip().startswith("|") and "\n|" in raw and "---" in raw:
        return raw

    sample = raw[:4096]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=[",", "\t", ";", "|"])
    except Exception:
        dialect = csv.excel

    try:
        reader = csv.reader(io.StringIO(raw), dialect)
        rows = [r for r in reader if any((c or "").strip() for c in r)]
    except Exception:
        return raw

    if not rows:
        return raw

    widest = max((len(r) for r in rows), default=0)
    if widest <= 1 and len(rows) <= 1:
        return raw

    col_count = min(widest, max_cols)
    norm_rows = []
    for r in rows:
        rr = (r + [""] * col_count)[:col_count]
        norm_rows.append(rr)

    def _cell(s: str) -> str:
        s = ("" if s is None else str(s)).replace("\n", " ").strip()
        s = s.replace("|", "\\|")
        if len(s) > max_cell_chars:
            s = s[: max_cell_chars - 1] + "…"
        return s

    norm_rows = [[_cell(c) for c in r] for r in norm_rows]

    header = norm_rows[0]
    body = norm_rows[1 : 1 + max_rows]

    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * col_count) + " |",
    ]
    for r in body:
        lines.append("| " + " | ".join(r) + " |")

    remaining = max(0, (len(norm_rows) - 1) - len(body))
    if remaining:
        lines.append("")
        lines.append(f"_… {remaining} more rows omitted …_")

    return "\n".join(lines)


class ArticleGenerationModule(dspy.Module):
    """Use the information collected from the information-seeking conversation to write a section."""

    def __init__(
        self,
        engine: Union[dspy.dsp.LM, dspy.dsp.HFModel],
    ):
        super().__init__()
        self.write_section = dspy.Predict(WriteSection)
        self.engine = engine

    def _get_cited_information_string(
        self,
        all_citation_index: Set[int],
        knowledge_base: KnowledgeBase,
        max_words: int = 1500,
    ):
        information = []
        cur_word_count = 0
        for index in sorted(list(all_citation_index)):
            info = knowledge_base.info_uuid_to_info_dict[index]
            snippet = info.snippets[0]
            info_text = f"[{index}]: {snippet}" + \
    (f" (Question: {info.meta['question']})" if 'question' in info.meta else '') + \
    (f" (Query: {info.meta['query']})" if 'query' in info.meta else '')
            cur_snippet_length = len(info_text.split())
            if cur_snippet_length + cur_word_count > max_words:
                break
            cur_word_count += cur_snippet_length
            information.append(info_text)
        return "\n".join(information)

    def gen_section(
        self, topic: str, node: KnowledgeNode, knowledge_base: KnowledgeBase
    ):
        if node is None or len(node.content) == 0:
            return ""
        if (
            node.synthesize_output is not None
            and node.synthesize_output
            and not node.need_regenerate_synthesize_output
        ):
            return node.synthesize_output
        
        # Guarantee that one index will strictly appear in one section
        all_citation_index = set(node.content)
        information = self._get_cited_information_string(
            all_citation_index=all_citation_index, knowledge_base=knowledge_base
        )
        with dspy.settings.context(lm=self.engine):
            synthesize_output = clean_up_section(
                self.write_section(
                    topic=topic, info=information, section=node.name
                ).output
            )

        # Include tables from SQL
        sql_tables = []
        table_citations = {}

        for index in sorted(list(all_citation_index)):
            info = knowledge_base.info_uuid_to_info_dict[index]
            csv_host = os.getenv("DATATALK_CSV_HOST", "")
            if hasattr(info, 'url') and info.url and csv_host and csv_host in info.url:
                url = info.url
                # datatalk server
                viewer_prefix = os.getenv("DATATALK_VIEWER_URL", "")
                if viewer_prefix and viewer_prefix in url:
                    url = url.replace(viewer_prefix, '')
                    response = requests.get(url)
                    response.raise_for_status()  # Raise an error for bad responses
                    data = response.json()

                    # Check for visualization in meta["html_file_path"] first (could be HTML or PNG)
                    visualization_url = None
                    if hasattr(info, 'meta') and info.meta and 'html_file_path' in info.meta and info.meta['html_file_path']:
                        visualization_url = info.meta['html_file_path']
                    else:
                        # Fallback: try to find .html version by URL pattern
                        html_url = url.rsplit('.', 1)[0] + '.html'
                        try:
                            html_response = requests.get(html_url)
                            if html_response.status_code == 200:
                                visualization_url = html_url
                        except Exception as e:
                            print(f"Visualization not found: {e} at {html_url}")

                    if 'sql_result' in data and data['sql_result'] and data['sql_result'] != "":
                        # Append visualization link if exists
                        if visualization_url:
                            data['sql_result'] += f"\n\n[click here for visualization]({visualization_url})"
                        table_citations[index] = data['sql_result']
                        sql_tables.append(data['sql_result'])
                # decomposition server
                else:
                    response = requests.get(url)
                    response.raise_for_status()  # Raise an error for bad responses
                    data = {
                        "sql_result": _csv_text_to_markdown_table(response.text)
                    }

                    # Check for visualization in meta["html_file_path"] first (could be HTML or PNG)
                    visualization_url = None
                    if hasattr(info, 'meta') and info.meta and 'html_file_path' in info.meta and info.meta['html_file_path']:
                        visualization_url = info.meta['html_file_path']
                    else:
                        # Fallback: Check for .html version, then .png version if not found
                        html_url = url.rsplit('.', 1)[0] + '.html'
                        png_url = url.rsplit('.', 1)[0] + '.png'
                        try:
                            html_response = requests.get(html_url)
                            if html_response.status_code == 200:
                                visualization_url = html_url
                            else:
                                # Try .png if .html does not exist
                                png_response = requests.get(png_url)
                                if png_response.status_code == 200:
                                    visualization_url = png_url
                        except Exception as e:
                            print(f"Visualization not found: {e} at {html_url} or {png_url}")

                    if 'sql_result' in data and data['sql_result'] and data['sql_result'] != "":
                        # Append visualization link if exists
                        if visualization_url:
                            data['sql_result'] += f"\n[click here for visualization]({visualization_url})"
                        table_citations[index] = data['sql_result']
                        sql_tables.append(data['sql_result'])
                    

        # Store the table information at the node level for post-processing
        node.table_citations = table_citations
        node.sql_tables = sql_tables

        node.synthesize_output = synthesize_output
        node.need_regenerate_synthesize_output = False
        return node.synthesize_output

    def forward(self, knowledge_base: KnowledgeBase):
        all_nodes = knowledge_base.collect_all_nodes()
        node_to_paragraph = {}

        # Define a function to generate paragraphs for nodes
        def _node_generate_paragraph(node):
            node_gen_paragraph = self.gen_section(
                topic=knowledge_base.topic, node=node, knowledge_base=knowledge_base
            )
            lines = node_gen_paragraph.split("\n")
            if lines[0].strip().replace("*", "").replace("#", "") == node.name:
                lines = lines[1:]
            node_gen_paragraph = "\n".join(lines)
            path = " -> ".join(node.get_path_from_root())
            return path, node_gen_paragraph

        with ThreadPoolExecutor(max_workers=5) as executor:
            # Submit all tasks
            future_to_node = {
                executor.submit(_node_generate_paragraph, node): node
                for node in all_nodes
            }

            # Collect the results as they complete
            for future in as_completed(future_to_node):
                path, node_gen_paragraph = future.result()
                node_to_paragraph[path] = node_gen_paragraph

        # Create a map of all tables in the article, in order of appearance
        table_map = {}
        table_count = 0

        # First get an ordered list of nodes as they'll appear in the final article
        ordered_nodes = []

        def collect_ordered_nodes(cur_root):
            if cur_root is not None:
                ordered_nodes.append(cur_root)
                for child in cur_root.children:
                    collect_ordered_nodes(child)
        
        for child in knowledge_base.root.children:
            collect_ordered_nodes(child)

        # Now number the tables in order of article appearance
        for node in ordered_nodes:
            if hasattr(node, 'table_citations') and node.table_citations:
                for citation_index in node.table_citations:
                    table_content = node.table_citations[citation_index]
                    if table_content and table_content != "":
                        table_count += 1
                        node_path = " -> ".join(node.get_path_from_root())
                        table_map[f"{node_path}_{citation_index}"] = table_count
     
        # Function to update node paragraphs with table references and numbers
        def update_paragraph_with_tables(node, paragraph):
            if not hasattr(node, 'table_citations') or not node.table_citations:
                return paragraph
                
            updated_paragraph = paragraph
            
            # Replace citations with table references
            for citation_index, table_content in node.table_citations.items():
                if not table_content or table_content == "":
                    continue
                
                node_path = " -> ".join(node.get_path_from_root())
                table_num = table_map.get(f"{node_path}_{citation_index}")
                if table_num:
                    updated_paragraph = updated_paragraph.replace(
                        f"[{citation_index}]", f"(Table {table_num})"
                    )
                    updated_paragraph += f"\n\n**Table {table_num}** [{citation_index}]\n{table_content}"
                    
            return updated_paragraph
       
        # Update all node paragraphs with proper table references
        for path, paragraph in node_to_paragraph.items():
            # Find the corresponding node
            for node in all_nodes:
                if " -> ".join(node.get_path_from_root()) == path:
                    node_to_paragraph[path] = update_paragraph_with_tables(node, paragraph)
                    break

        def helper(cur_root, level):
            to_return = []
            if cur_root is not None:
                hash_tag = "#" * level + " "
                cur_path = " -> ".join(cur_root.get_path_from_root())
                node_gen_paragraph = node_to_paragraph[cur_path]
                to_return.append(f"{hash_tag}{cur_root.name}\n{node_gen_paragraph}")
                for child in cur_root.children:
                    to_return.extend(helper(child, level + 1))
            return to_return

        to_return = []
        for child in knowledge_base.root.children:
            to_return.extend(helper(child, level=1))

        return "\n".join(to_return)


class WriteSection(dspy.Signature):
    """Write a Wikipedia section based on the collected information. You will be given the topic, the section you are writing and relevant information.
    Each information will be provided with the raw content along with question and query lead to that information.
    Here is the format of your writing:
    Use [1], [2], ..., [n] in line (for example, "The capital of the United States is Washington, D.C.[1][3]."). You DO NOT need to include a References or Sources section to list the sources at the end.
    """

    info = dspy.InputField(prefix="The collected information:\n", format=str)
    topic = dspy.InputField(prefix="The topic of the page: ", format=str)
    section = dspy.InputField(prefix="The section you need to write: ", format=str)
    output = dspy.OutputField(
        prefix="Write the section with proper inline citations (Start your writing. Don't include the page title, section name, or try to write other sections. Do not start the section with topic name.):\n",
        format=str,
    )
