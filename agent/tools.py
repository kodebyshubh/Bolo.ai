"""
Mock tool(s) for the voice agent. get_order_status is a fake in-memory
lookup -- no real backend, just enough to demonstrate tool calling.
"""

ORDERS = {
    "12345": {"status": "shipped", "expected_delivery": "2026-09-20"},
    "67890": {"status": "processing", "expected_delivery": "2026-09-25"},
    "54321": {"status": "delivered", "expected_delivery": "2026-09-10"},
}


def get_order_status(order_id: str) -> dict:
    order = ORDERS.get(order_id)
    if order is None:
        return {"error": f"No order found with id {order_id}"}
    return {"order_id": order_id, **order}


# OpenAI/Groq-compatible tool schema, registered with the LLM in agent/llm.py
TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "get_order_status",
            "description": "Look up the shipping status and expected delivery date for a customer's order by its order ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "The order ID to look up, e.g. '12345'.",
                    },
                },
                "required": ["order_id"],
            },
        },
    },
]

AVAILABLE_FUNCTIONS = {
    "get_order_status": get_order_status,
}
