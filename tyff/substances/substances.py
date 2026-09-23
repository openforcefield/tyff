import pydantic

from tyff.substances.component import Component


class Substance(pydantic.BaseModel):
    components: tuple[Component, ...] = pydantic.Field(default_factory=tuple)
    amounts: dict[Component, float] = pydantic.Field(default_factory=dict)

    def add_component(self, component: Component, amount: float):
        """Adds a component to the substance with the specified amount.

        Parameters
        ----------
        component: Component
            The component to add.
        amount: float
            The amount of the component to add. This should be in units of moles.

        Returns
        -------
        None
        """
        if component not in self.components:
            self.components += (component,)

        self.amounts[component] = self.amounts.get(component, 0.0) + amount
