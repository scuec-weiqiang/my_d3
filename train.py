import d3rlpy

dataset, env = d3rlpy.datasets.get_pendulum()
# setup CQL algorithm
cql = d3rlpy.algos.CQLConfig().create(device='cuda:0')
cql.build_with_dataset(dataset)

# start training
cql.fit(
    dataset,
    n_steps=100000,
    n_steps_per_epoch=10000,
    evaluators={
        'environment': d3rlpy.metrics.EnvironmentEvaluator(env), # evaluate with CartPole-v1 environment
    },
)

