import asyncio

async def stage2_collect_rankings(query, stage1_results):
    print(f"[Mock Council] stage2_collect_rankings")
    results = [
        {"model": res["model"], "expected_improvement": 0.05, "probability_of_success": 0.8, "implementation_cost": "0.1 gpu hours"}
        for res in stage1_results
    ]
    return results, {res["model"]: res["model"] for res in stage1_results}

def calculate_aggregate_rankings(stage2_results, label_to_model):
    return stage2_results
