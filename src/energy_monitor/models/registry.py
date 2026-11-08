from collections.abc import Sequence
from typing import Any


class ModelRegistry[ModelType]:
    """Registro por nome de um tipo de modelo.

    Experimentos e API localizam os modelos por este registro; acrescentar uma
    técnica exige apenas criar a classe e registrá-la com `register`.
    """

    def __init__(self, kind: str) -> None:
        """Cria um registro vazio para o tipo `kind`."""
        self.kind = kind
        self._models: dict[str, type[ModelType]] = {}

    def register(self, model_class: type[ModelType]) -> type[ModelType]:
        """Registra `model_class` pelo seu atributo `name`; pode ser decorador."""
        name = model_class.name
        if name in self._models:
            raise ValueError(f"Modelo de {self.kind} já registrado: {name}")
        self._models[name] = model_class
        return model_class

    def unregister(self, name: str) -> None:
        """Remove o modelo `name` do registro."""
        del self._models[name]

    def get(self, name: str) -> type[ModelType]:
        """Retorna a classe registrada como `name`."""
        if name not in self._models:
            raise KeyError(f"Modelo de {self.kind} desconhecido: {name}")
        return self._models[name]

    def names(self) -> list[str]:
        """Lista os nomes registrados em ordem alfabética."""
        return sorted(self._models)

    def create(
        self, name: str, feature_columns: Sequence[str], seed: int, **hyperparameters: Any
    ) -> ModelType:
        """Instancia o modelo `name`."""
        return self.get(name)(feature_columns, seed=seed, **hyperparameters)
