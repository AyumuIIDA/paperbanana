import asyncio
from pathlib import Path
from paperbanana import PaperBananaPipeline, GenerationInput, DiagramType
from paperbanana.core.config import Settings

def main():
    md_path = Path("../graph.md")
    source_text = md_path.read_text(encoding="utf-8")

    settings = Settings(
        VLM_PROVIDER="gemini",
        VLM_MODEL="gemini-3-pro-image-preview",
        IMAGE_PROVIDER="google_imagen",
        IMAGE_MODEL="gemini-3-pro-image-preview",
        optimize_inputs=False,
        auto_refine=False,
    )

    pipeline = PaperBananaPipeline(settings=settings)
    result = asyncio.run(
        pipeline.generate(
            GenerationInput(
                source_context=source_text,
                communicative_intent="RAGの定式化 p(y|x,D) と、クラウドRAGにおける主要コスト要素（インデックス構築、検索、LLMトークン推論）の対応関係を示す。図中ラベルは日本語で表記する。",
                diagram_type=DiagramType.METHODOLOGY,
            )
        )
    )

    print("Saved to: ",result.image_path)

if __name__ == "__main__":
    main()