from sonic.computer_use.action_contracts import (
    ActionContract,
    ActionContractValidator,
    ActionReceipt,
    StateDelta,
)
from sonic.computer_use.models import ActionExecutionStatus
from sonic.safety.kernel import KernelVerdict, SafetyAuthorization


class _AllowKernel:
    def __init__(self, verdict=KernelVerdict.ALLOW):
        self.verdict = verdict
        self.calls = 0

    def authorize(self, action_type, parameters, provider=None):
        self.calls += 1
        return SafetyAuthorization(
            verdict=self.verdict,
            reason="test",
            action_type=action_type,
        )


def test_contract_validation_reuses_safety_and_checks_preconditions():
    kernel = _AllowKernel()
    validator = ActionContractValidator(safety_kernel=kernel)
    contract = ActionContract("ACTION", preconditions={"mode": "ready"})

    result = validator.validate(contract, state={"mode": "ready"}, parameters={"x": 1})

    assert result.valid
    assert kernel.calls == 1


def test_validator_fails_closed_for_safety_and_precondition_failures():
    kernel = _AllowKernel(KernelVerdict.DENY)
    validator = ActionContractValidator(safety_kernel=kernel)
    contract = ActionContract("ACTION", preconditions={"mode": "ready"})

    result = validator.validate(contract, state={"mode": "blocked"})

    assert not result.valid
    assert any(reason.startswith("safety:") for reason in result.reasons)
    assert any(reason.startswith("precondition") for reason in result.reasons)


def test_receipt_requires_real_delta_preconditions_and_postconditions():
    contract = ActionContract(
        "ACTION",
        preconditions={"mode": "ready"},
        postconditions={"mode": "done"},
    )
    before = {"mode": "ready"}
    after = {"mode": "done", "count": 1}
    receipt = ActionReceipt(
        action_type="ACTION",
        status=ActionExecutionStatus.SUCCEEDED,
        before=before,
        after=after,
        state_delta=StateDelta.from_states(before, after),
        exit_code=0,
    )

    assert ActionContractValidator(safety_kernel=_AllowKernel()).validate_receipt(
        contract, receipt
    ).valid

    forged = ActionReceipt(
        action_type="ACTION",
        status=ActionExecutionStatus.SUCCEEDED,
        before=before,
        after=after,
        state_delta=StateDelta(),
        exit_code=0,
    )
    assert not ActionContractValidator(safety_kernel=_AllowKernel()).validate_receipt(
        contract, forged
    ).valid


def test_receipt_without_execution_evidence_or_matching_precondition_is_rejected():
    contract = ActionContract(
        "ACTION",
        preconditions={"mode": "ready"},
        postconditions={"mode": "done"},
    )
    before = {"mode": "blocked"}
    after = {"mode": "done"}
    receipt = ActionReceipt(
        action_type="ACTION",
        status=ActionExecutionStatus.SUCCEEDED,
        before=before,
        after=after,
        state_delta=StateDelta.from_states(before, after),
    )
    result = ActionContractValidator(safety_kernel=_AllowKernel()).validate_receipt(
        contract, receipt
    )
    assert not result.valid
    assert "neither a successful exit code nor evidence" in " ".join(result.reasons)
    assert "precondition failed in receipt: mode" in result.reasons


def test_failed_receipt_cannot_satisfy_postconditions():
    contract = ActionContract("ACTION", postconditions={"value": 1})
    receipt = ActionReceipt(
        action_type="ACTION",
        status=ActionExecutionStatus.FAILED,
        before={"value": 0},
        after={"value": 1},
        state_delta=StateDelta.from_states({"value": 0}, {"value": 1}),
    )

    result = ActionContractValidator(safety_kernel=_AllowKernel()).validate_receipt(
        contract, receipt
    )
    assert not result.valid
