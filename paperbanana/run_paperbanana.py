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
                communicative_intent='RAGの確率モデル(p_η, p_θ,D)と、クラウド型RAGにおける三層のコスト構造(インデックス維持費・検索課金・LLMトークン課金)の対応関係を示す概念図',
                diagram_type=DiagramType.METHODOLOGY,
            )
        )
    )

    print("Saved to: ",result.image_path)

if __name__ == "__main__":
    main()