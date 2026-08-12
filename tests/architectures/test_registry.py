from reclab.architectures import REGISTRY, Architecture, ArchitectureInfo


def test_every_registered_architecture_is_an_architecture_subclass():
    for arch_cls in REGISTRY.values():
        assert issubclass(arch_cls, Architecture)


def test_every_architecture_exposes_valid_static_info():
    for name, arch_cls in REGISTRY.items():
        info = arch_cls.info()
        assert isinstance(info, ArchitectureInfo)
        assert info.name == name
        assert info.description
        assert info.strengths
        assert info.weaknesses
        assert info.relative_train_cost in {"low", "medium", "high"}
        assert info.relative_serving_latency in {"low", "medium", "high"}
