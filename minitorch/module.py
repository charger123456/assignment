from _future_ import annotations
from typing import Any, Dict, Optional, Sequence, Tuple

class Module:
    """Modules form a tree that store parameters and other
    submodules. They make up the basis of neural network stacks.

    Attributes
    ----------
        _modules : Storage of the child modules
        _parameters : Storage of the module's parameters
        training : Whether the module is in training mode or evaluation mode
    """

    _modules: Dict[str, Module]
    _parameters: Dict[str, Parameter]
    training: bool

    def _init_(self) -> None:
        self._modules = {}
        self._parameters = {}
        self.training = True

    def modules(self) -> Sequence[Module]:
        """Return the direct child modules of this module."""
        m: Dict[str, Module] = self._dict_["_modules"]
        return list(m.values())

    def train(self) -> None:
        """Set the mode of this module and all descendent modules to train."""
        self.training = True
        for module in self.modules():
            module.train()

    def eval(self) -> None:
        """Set the mode of this module and all descendent modules to eval."""
        self.training = False
        for module in self.modules():
            module.eval()

    def named_parameters(self) -> Sequence[Tuple[str, Parameter]]:
        """Collect all the parameters of this module and its descendants.

        Returns
        -------
            The name and Parameter of each ancestor parameter.
        """
        result = []
        for name, param in self._parameters.items():
            result.append((name, param))
        
        for name, module in self._modules.items():
            result.extend((f"{name}.{param_name}", param)
                          for param_name, param in module.named_parameters())

        return result

    def parameters(self) -> Sequence[Parameter]:
        """Enumerate over all the parameters of this module and its descendants."""
        result = list(self._parameters.values())
        
        for module in self._modules.values():
            result.extend(module.parameters())
        
        return result

    def add_parameter(self, k: str, v: Any) -> Parameter:
        """Manually add a parameter. Useful helper for scalar parameters.

        Args:
        ----
            k: Local name of the parameter.
            v: Value for the parameter.

        Returns
        -------
            Newly created parameter.
        """
        val = Parameter(v, k)
        self._dict_["_parameters"][k] = val
        return val

    def _setattr_(self, key: str, val: Parameter) -> None:
        if isinstance(val, Parameter):
            self._dict_["_parameters"][key] = val
        elif isinstance(val, Module):
            self._dict_["_modules"][key] = val
        else:
            super()._setattr_(key, val)

    def _getattr_(self, key: str) -> Any:
        if key in self._dict_["_parameters"]:
            return self._dict_["_parameters"][key]

        if key in self._dict_["_modules"]:
            return self._dict_["_modules"][key]
        return None

    def _call_(self, *args: Any, **kwargs: Any) -> Any:
        return self.forward(*args, **kwargs)

    def _repr_(self) -> str:
        def addindent(s: str, numSpaces: int) -> str:
            s2 = s_.split("\n")
            if len(s2) == 1:
                return s_
            first = s2.pop(0)
            s2 = [(numSpaces * " ") + line for line in s2]
            s = "\n".join(s2)
            s = first + "\n" + s
            return s

        child_lines = []

        for key, module in self._modules.items():
            mod_str = repr(module)
            mod_str = _addindent(mod_str, 2)
            child_lines.append("(" + key + "): " + mod_str)
        lines = child_lines

        main_str = self._class.name_ + "("
        if lines:
            # simple one-liner info, which most builtin Modules will use
            main_str += "\n  " + "\n  ".join(lines) + "\n"

        main_str += ")"
        return main_str


class Parameter:
    """A Parameter is a special container stored in a Module.

    It is designed to hold a Variable, but we allow it to hold
    any value for testing.
    """

    def _init_(self, x: Any, name: Optional[str] = None) -> None:
        self.value = x
        self.name = name
        if hasattr(x, "requires_grad_"):
            self.value.requires_grad_(True)
            if self.name:
                self.value.name = self.name

    def update(self, x: Any) -> None:
        """Update the parameter value."""
        self.value = x
        if hasattr(x, "requires_grad_"):
            self.value.requires_grad_(True)
            if self.name:
                self.value.name = self.name

    def _repr_(self) -> str:
        return repr(self.value)

    def _str_(self) -> str:
        return str(self.value)
