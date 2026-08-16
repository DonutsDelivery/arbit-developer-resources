#pragma once

#include "visual_plan_executor.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace videowire
{
struct VisualPlanExecutionSnapshot
{
    std::vector<CompiledVisualLayerPlan> plans;
    VisualPlanExecutionState state;
};

inline bool makeVisualPlanExecutionSnapshot (
    std::vector<CompiledVisualLayerPlan> plans,
    std::shared_ptr<VisualPlanExecutionSnapshot>& snapshot,
    std::string& diagnostic, VisualPlanTelemetry* telemetryOwner = nullptr)
{
    auto candidate = std::make_shared<VisualPlanExecutionSnapshot>();
    if (telemetryOwner != nullptr) candidate->state.setTelemetryOwner(*telemetryOwner);
    candidate->plans = std::move(plans);
    diagnostic.clear();
    if (! candidate->state.admitPlans(candidate->plans, &diagnostic))
        return false;

    std::vector<VisualTelemetryPlanAdmission> telemetryPlans;
    telemetryPlans.reserve(candidate->plans.size());
    for (const auto& plan : candidate->plans)
        telemetryPlans.push_back(makeVisualTelemetryAdmission(plan));
    candidate->state.telemetry().admitPlans(telemetryPlans);
    snapshot = std::move(candidate);
    return true;
}
} // namespace videowire
