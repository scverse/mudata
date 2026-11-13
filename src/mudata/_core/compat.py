try:
    from anndata.acc import (
        AdRef,
        GraphAcc,
        GraphMapAcc,
        LayerAcc,
        LayerMapAcc,
        MapAcc,
        MetaAcc,
        MultiAcc,
        MultiMapAcc,
        RefAcc,
    )

    from ..acc import (
        ModAcc,
        ModGraphAcc,
        ModGraphMapAcc,
        ModLayerAcc,
        ModLayerMapAcc,
        ModMapAcc,
        ModMetaAcc,
        ModMultiAcc,
        ModMultiMapAcc,
        MultiModAcc,
        _ModalityAcc,
        _ModalityMixin,
    )
except ImportError:
    AdRef = type("AdRef", (), {"__module__": "anndata.acc"})  # type: ignore[assignment,misc]
    GraphAcc = type("GraphAcc", (), {"__module__": "anndata.acc"})  # type: ignore[assignment,misc]
    GraphMapAcc = type("GraphMapAcc", (), {"__module__": "anndata.acc"})  # type: ignore[assignment,misc]
    LayerAcc = type("LayerAcc", (), {"__module__": "anndata.acc"})  # type: ignore[assignment,misc]
    LayerMapAcc = type("LayerMapAcc", (), {"__module__": "anndata.acc"})  # type: ignore[assignment,misc]
    MetaAcc = type("MetaAcc", (), {"__module__": "anndata.acc"})  # type: ignore[assignment,misc]
    MultiAcc = type("MultiAcc", (), {"__module__": "anndata.acc"})  # type: ignore[assignment,misc]
    MultiMapAcc = type("MultiMapAcc", (), {"__module__": "anndata.acc"})  # type: ignore[assignment,misc]
    MapAcc = type("MapAcc", (), {"__module__": "anndata.acc"})  # type: ignore[assignment,misc]
    RefAcc = type("RefAcc", (), {"__module__": "anndata.acc"})  # type: ignore[assignment,misc]

    MultiModAcc = type("MultiModAcc", (), {"__module__": "mudata.acc"})  # type: ignore[assignment,misc]
    ModLayerAcc = type("ModLayerAcc", (), {"__module__": "mudata.acc"})  # type: ignore[assignment,misc]
    ModLayerMapAcc = type("ModLayerMapAcc", (), {"__module__": "mudata.acc"})  # type: ignore[assignment,misc]
    ModMetaAcc = type("ModMetaAcc", (), {"__module__": "mudata.acc"})  # type: ignore[assignment,misc]
    ModMultiAcc = type("ModMultiAcc", (), {"__module__": "mudata.acc"})  # type: ignore[assignment,misc]
    ModMultiMapAcc = type("ModMultiMapAcc", (), {"__module__": "mudata.acc"})  # type: ignore[assignment,misc]
    ModGraphAcc = type("ModGraphAcc", (), {"__module__": "mudata.acc"})  # type: ignore[assignment,misc]
    ModGraphMapAcc = type("ModGraphMapAcc", (), {"__module__": "mudata.acc"})  # type: ignore[assignment,misc]
    ModMapAcc = type("ModMapAcc", (), {"__module__": "mudata.acc"})  # type: ignore[assignment,misc]
    ModAcc = type("ModAcc", (), {"__module__": "mudata.acc"})  # type: ignore[assignment,misc]
    _ModalityAcc = type("_ModalityAcc", (), {"__module__": "mudata.acc"})  # type: ignore[assignment,misc]
    _ModalityMixin = type("_ModalityMixin", (), {"__module__": "mudata.acc"})  # type: ignore[assignment,misc]
