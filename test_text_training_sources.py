from __future__ import annotations

from pathlib import Path

from custom_llm import (
    LogicAwareTokenizer,
    collect_wikipedia_article_texts,
    logic_formula_prompt,
    read_plain_text_files,
    read_wikipedia_titles_file,
    statement_prompt,
    wikipedia_title_from_line,
)
from logic import Imp, Var
from statement_generation import LabeledStatement


def test_default_wikipedia_title_file_has_500_unique_titles() -> None:
    titles = read_wikipedia_titles_file(Path("data") / "wikipedia_titles_500.txt")

    assert len(titles) == 500
    assert len(set(titles)) == 500
    assert all(title.strip() == title for title in titles)


def test_wikipedia_title_lines_accept_titles_and_urls() -> None:
    assert wikipedia_title_from_line("Truth table") == "Truth table"
    assert wikipedia_title_from_line("https://en.wikipedia.org/wiki/First-order_logic") == "First-order logic"
    assert wikipedia_title_from_line("  ") == ""


def test_plain_text_file_reader_skips_empty_files() -> None:
    text_path = Path("_plain_text_source_test.txt")
    empty_path = Path("_empty_text_source_test.txt")
    try:
        text_path.write_text("Logic is the study of valid inference.\n", encoding="utf-8")
        empty_path.write_text("   \n", encoding="utf-8")

        assert read_plain_text_files([text_path, empty_path]) == [
            "Logic is the study of valid inference.\n"
        ]
    finally:
        text_path.unlink(missing_ok=True)
        empty_path.unlink(missing_ok=True)


def test_wikipedia_collection_can_use_fake_fetcher() -> None:
    titles_path = Path("_wiki_title_source_test.txt")
    try:
        titles_path.write_text(
            "# comment\n"
            "Propositional calculus\n"
            "https://en.wikipedia.org/wiki/Truth_table\n",
            encoding="utf-8",
        )

        def fake_fetcher(title: str) -> str:
            return f"{title} article text with enough characters for training."

        texts = collect_wikipedia_article_texts(
            titles_path=titles_path,
            min_chars=10,
            fetcher=fake_fetcher,
        )

        assert len(texts) == 2
        assert texts[0].startswith("Title: Propositional calculus\n")
        assert texts[1].startswith("Title: Truth table\n")
    finally:
        titles_path.unlink(missing_ok=True)


def test_statement_prompt_can_use_symbolic_label_override() -> None:
    statement = LabeledStatement(
        name="bad_saved_label",
        formula=Imp(Var("P"), Var("P")),
        label=False,
    )

    prompt = statement_prompt(statement, include_label=True, label_override=True)

    assert prompt.endswith("Answer: true")


def test_statement_prompt_without_label_hides_saved_label_metadata() -> None:
    statement = LabeledStatement(
        name="true_looks_like_label",
        formula=Imp(Var("P"), Var("P")),
        label=True,
    )

    prompt = statement_prompt(statement, include_label=False).lower()

    assert "true" not in prompt
    assert "false" not in prompt
    assert statement.name not in prompt


def test_logic_formula_prompt_tokenizes_formula_once() -> None:
    tokenizer = LogicAwareTokenizer()
    tokens = tokenizer.normalize(logic_formula_prompt(Imp(Var("P"), Var("Q"))))

    assert "VAR_P" in tokens
    assert "VAR_Q" in tokens
    assert "VAR_VAR_P" not in tokens
    assert "<TRUE>" not in tokens
    assert "<FALSE>" not in tokens


def test_logic_formula_prompt_tokenizes_word_variables_as_variables() -> None:
    tokenizer = LogicAwareTokenizer()
    tokens = tokenizer.normalize(logic_formula_prompt(Imp(Var("active"), Var("atLeastOne"))))

    assert "VAR_active" in tokens
    assert "VAR_atLeastOne" in tokens


def test_logic_formula_prompt_keeps_reserved_words_as_formula_variables() -> None:
    tokenizer = LogicAwareTokenizer()
    tokens = tokenizer.normalize(logic_formula_prompt(Imp(Var("true"), Var("statement"))))

    assert "VAR_true" in tokens
    assert "VAR_statement" in tokens
    assert "<TRUE>" not in tokens

    answer_tokens = tokenizer.normalize(" true false")
    assert answer_tokens == ["<TRUE>", "<FALSE>"]


if __name__ == "__main__":
    test_default_wikipedia_title_file_has_500_unique_titles()
    test_wikipedia_title_lines_accept_titles_and_urls()
    test_plain_text_file_reader_skips_empty_files()
    test_wikipedia_collection_can_use_fake_fetcher()
    test_statement_prompt_can_use_symbolic_label_override()
    test_statement_prompt_without_label_hides_saved_label_metadata()
    test_logic_formula_prompt_tokenizes_formula_once()
    test_logic_formula_prompt_tokenizes_word_variables_as_variables()
    test_logic_formula_prompt_keeps_reserved_words_as_formula_variables()
    print("text training source checks passed")
