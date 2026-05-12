"""Product recommendation tool — agent recommends gear/equipment/nutrition.

The agent decides WHAT to recommend (product name, category, reason).
This tool enriches those recommendations with real product data
(image, price, URL) via the ProductSearchService, decorates URLs with
affiliate links, and persists everything to the DB.

The app then reads product_recommendations from Supabase and renders
a horizontal product bar.
"""

from __future__ import annotations

import logging

from src.agent.tools.registry import Tool, ToolRegistry
from src.config import get_settings

logger = logging.getLogger(__name__)


def register_product_tools(registry: ToolRegistry, user_model) -> None:
    """Register product recommendation tools on the given *registry*."""
    settings = get_settings()

    def recommend_products(
        products: list[dict],
        session_id: str | None = None,
        plan_id: str | None = None,
    ) -> dict:
        """Enrich, affiliatize, and save product recommendations.

        Args:
            products: List of dicts, each with keys:
                - name (str, required): Product name
                - category (str): e.g. "shoes", "watch", "recovery"
                - reason (str, required): Why this product fits the athlete
                - search_query (str): Query for product search API
            session_id: Optional training session ID to link to.
            plan_id: Optional training plan ID to link to.

        Returns:
            Summary dict with saved_count and product names.
        """
        import asyncio

        user_id = settings.agenticsports_user_id
        if not user_id:
            return {"error": "No user configured", "saved_count": 0}

        if not products:
            return {"error": "No products provided", "saved_count": 0}

        # Run the async enrichment pipeline synchronously
        # (agent tools run in sync context via asyncio.to_thread)
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We're inside an async context — use a helper
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    enriched = pool.submit(
                        asyncio.run,
                        _enrich_products(products, user_id, session_id, plan_id),
                    ).result()
            else:
                enriched = asyncio.run(
                    _enrich_products(products, user_id, session_id, plan_id),
                )
        except Exception as exc:
            logger.warning("Product enrichment failed, saving without enrichment: %s", exc)
            enriched = _fallback_products(products, user_id, session_id, plan_id)

        # Save to DB
        from src.db.product_recommendations_db import save_recommendations
        saved = save_recommendations(user_id, enriched)

        return {
            "saved_count": len(saved),
            "products": [
                {
                    "name": r.get("product_name", ""),
                    "category": r.get("category"),
                    "has_image": bool(r.get("image_url")),
                    "has_price": r.get("price") is not None,
                    "has_affiliate": bool(r.get("affiliate_url")),
                }
                for r in saved
            ],
        }

    # DEPRECATED: recommend_products - monetization stub, not active. NOT registered.
    # registry.register(Tool(
    #     name="recommend_products",
    #     description=(
    #         "Recommend 3-4 products (gear, equipment, nutrition, recovery) "
    #         "relevant to the athlete's training context."
    #     ),
    #     handler=recommend_products,
    #     parameters={
    #         "type": "object",
    #         "properties": {
    #             "products": {"type": "array", "items": {"type": "object"}},
    #         },
    #         "required": ["products"],
    #     },
    #     category="planning",
    # ))


async def _enrich_products(
    products: list[dict],
    user_id: str,
    session_id: str | None,
    plan_id: str | None,
) -> list[dict]:
    """Enrich product recommendations with search data and affiliate links.

    For each product:
    1. Search via ProductSearchService (image, price, URL)
    2. Affiliatize the URL
    3. Build the recommendation dict
    """
    from src.services.product_search import get_product_search_provider
    from src.services.affiliate_provider import affiliatize

    search_provider = get_product_search_provider()

    enriched = []
    for product in products:
        rec = _build_base_rec(product, user_id, session_id, plan_id)

        query = product.get("search_query") or product.get("name", "")
        category = product.get("category")

        # Try to enrich via search API
        if search_provider and query:
            try:
                results = await search_provider.search(
                    query=query,
                    category=category,
                    max_results=1,
                )
                if results:
                    best = results[0]
                    rec = {
                        **rec,
                        "product_name": best.name,
                        "image_url": best.image_url,
                        "price": best.price,
                        "currency": best.currency,
                        "product_url": best.product_url,
                        "source": best.source,
                    }
            except Exception as exc:
                logger.warning("Search failed for '%s': %s", query, exc)

        # Affiliatize the URL
        product_url = rec.get("product_url")
        if product_url:
            try:
                aff_url, provider_name = await affiliatize(product_url)
                if provider_name:
                    rec = {
                        **rec,
                        "affiliate_url": aff_url,
                        "affiliate_provider": provider_name,
                    }
            except Exception as exc:
                logger.warning("Affiliatize failed for '%s': %s", product_url, exc)

        enriched.append(rec)

    return enriched


def _build_base_rec(
    product: dict,
    user_id: str,
    session_id: str | None,
    plan_id: str | None,
) -> dict:
    """Build a base recommendation dict from agent-provided product data."""
    return {
        "user_id": user_id,
        "product_name": product.get("name", "Unknown Product"),
        "reason": product.get("reason", ""),
        "category": product.get("category"),
        "sport": product.get("sport"),
        "search_query": product.get("search_query") or product.get("name"),
        "source": "llm",
        "session_id": session_id,
        "plan_id": plan_id,
    }


def _fallback_products(
    products: list[dict],
    user_id: str,
    session_id: str | None,
    plan_id: str | None,
) -> list[dict]:
    """Build recommendation dicts without enrichment (fallback)."""
    return [
        _build_base_rec(p, user_id, session_id, plan_id)
        for p in products
    ]
