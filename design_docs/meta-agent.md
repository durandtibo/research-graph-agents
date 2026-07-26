# Meta-Agent

## Agent

- Update the agent
    - Parameters are variable
    - Maybe we can use a method with this signature `def update(self, params: dict) -> None:`
- Evaluate the performance of the agent
- Diagnostic the agent mistakes/errors
- Generate audit logs

What should be the definition of the AIAgent?

```python
class EvaluationState:
    input: T
    output: S
```

- Inference to generate the predictions
- Evaluation to evaluate the predictions
- Diagnostic to identify the main mistakes

Each "agent" can be a multi-agents

## Task

The task to use to evaluate the AI agent.
Can be seen as the inference pipeline + metrics + diagnostics.

I wonder if it is a better design to pass the agent to the task, or to pass the task to the agent.

## Meta-Agent

- Takes as input the history.
- The history can be represented as a sequence/list of dicts.
- Each dict has at least the following keys:
    - `iteration` (optional): contains the iteration value
    - `agent_hyperparams`: contains the value of the agent hyper-parameters i.e. the
      parameters that will need to be optimized for the task.
    - `metrics`: contains the metrics associated to the hyper-parameters.
    - `diagnostics`

The meta-agent should be independent of the task, so it should be dependent of the hyper-parameters
to optimize.
For example, we can have one meta-agent just to optimize the system prompt, another one to optimize
the system prompt and tool prompt.

The meta-agent is just a special agent that is used to optimize the hyper-parameters of another
agent.

Maybe it is a good idea to provide a list of guardrails that should be included in the generated
hyper-parameters.
The guardrails can be either passed explicitly to the system prompt of the meta-agent so they are
always append to the generated hyper-parameters.
