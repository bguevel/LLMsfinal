from __future__ import annotations

from pathlib import Path

from custom_llm import (
    collect_wikipedia_article_texts,
    read_plain_text_files,
    read_wikipedia_titles_file,
    wikipedia_title_from_line,
)


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


if __name__ == "__main__":
    test_default_wikipedia_title_file_has_500_unique_titles()
    test_wikipedia_title_lines_accept_titles_and_urls()
    test_plain_text_file_reader_skips_empty_files()
    test_wikipedia_collection_can_use_fake_fetcher()
    print("text training source checks passed")
