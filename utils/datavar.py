import pandas as pd

from .constants import SSP_SCENARIOS, YEARS, IP_SCENARIOS
from .data import get_interp


class Var:
    def __init__(
        self,
        data,
        scenarios,
        vetted_scenarios,
        variable=None,
        year=None,
        values=None,
        default=None,
    ):
        self.data = data
        self.scenarios = scenarios
        self.vetted_scenarios = vetted_scenarios

        if variable is None and values is None:
            raise Exception("variable and values cannot both be None")
        if variable is not None and values is not None:
            raise Exception("variable and values cannot both be defined")

        if variable is not None:
            self._variable = variable
            if year is None:
                year = YEARS
            to_series = not isinstance(year, (tuple, list))
            year = _to_list(year, to_str=True)
            self._year = year[0] if to_series else year

            existing_years = list(set(year).intersection(set(YEARS)))
            interp_years = list(set(year) - set(YEARS))

            # First get years that already exist
            self._values = data[data["Variable"] == variable].set_index("Name")[
                existing_years
            ]
            # Then interpolate all other years:
            for y in interp_years:
                self._values[y] = get_interp(data, None, [variable], float(y))[
                    0
                ].T.iloc[:, 0]
            self._values = self._values[self._year]

        if values is not None:
            if isinstance(values, pd.Series):
                self._year = str(values.name)
            else:
                self._year = list(values.columns)
            self._values = values

        self.default = default
        if self.default is not None:
            self._values = self._values.fillna(self.default)

    def select(
        self, category=None, ip=None, ssp=None, curpol=None, ndc=None, vetted=True
    ):
        """
        Filter the resulting values dataframe

        Choose between the following filters:
        - category: None, "all" or any/subset of [C1, C2, C3, C3, C4, C5, C6, C7, C8]
        - ip:       None, "all" or any/subset of [CurPol, ModAct, GS, Neg, Ren, LD, SP]
        - ssp:      None, "all" or any/subset of [SSP1-19, SSP1-26, SSP4-34, SSP2-45, SSP4-60, SSP3-70, SSP5-85]
        - curpol:   (not implemented)
        - ndc:      (not implemented)

        By default, only vetted scenarios are shown. Use vetted=False to include all scenarios
        """
        selection = (self.vetted_scenarios if vetted else self.scenarios).copy()
        extra_columns = []

        # Climate category
        all_categories = ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8"]
        if category is not None:
            if category == "all":
                category = all_categories
            category = _to_list(category)
            for c in category:
                if c not in all_categories:
                    raise KeyError(f"{c} is not a valid climate category.")
            selection = selection[selection["Category"].isin(_to_list(category))]
            extra_columns.append("Category")

        # Illustrative Pathways
        if ip is not None:
            if ip == "all":
                ip = list(IP_SCENARIOS.keys())
            ip = _to_list(ip)
            for v in ip:
                if v not in IP_SCENARIOS:
                    raise KeyError(
                        f"{v} is not a valid IP [{', '.join(IP_SCENARIOS.keys())}]"
                    )
            selection = selection[
                selection.index.isin([IP_SCENARIOS[v].scenario for v in ip])
            ]
            extra_columns.append("IP")

        # SSP
        if ssp is not None:
            if ssp == "all":
                ssp = list(SSP_SCENARIOS.keys())
            # Add SSPs from unvetted scenarios, since
            # SSP1-19, SSP4-34, SSP4-60 and SSP3-70 do not pass vetting
            ssp = _to_list(ssp)
            for v in ssp:
                if v not in SSP_SCENARIOS:
                    raise KeyError(
                        f"{v} is not a valid SSP [{', '.join(SSP_SCENARIOS.keys())}]"
                    )
            selection = selection[
                selection.index.isin([SSP_SCENARIOS[v].scenario for v in ssp])
            ]
            for v in ssp:
                if SSP_SCENARIOS[v].scenario not in selection.index:
                    # Add unvetted ssp
                    selection = selection.append(
                        self.scenarios.loc[SSP_SCENARIOS[v].scenario]
                    )
            extra_columns.append("SSP")

        # CurPol
        if curpol is not None:
            raise NotImplementedError("CurPol selection filter not implemented yet")

        # NDC
        if ndc is not None:
            raise NotImplementedError("NDC selection filter not implemented yet")

        subset_values = self._values[self._values.index.isin(selection.index)]
        is_series = isinstance(subset_values, pd.Series)
        if is_series:
            subset_values = subset_values.to_frame("value")
        # Merge with extra columns
        subset_values = (
            subset_values.merge(
                selection[extra_columns], left_index=True, right_index=True, how="left"
            )
            .reset_index()
            .sort_values(extra_columns)
            .set_index(extra_columns + ["Name"])
        )
        if is_series:
            # Change back to Series
            return subset_values["value"]
        return subset_values

    def _repr_html_(self):
        n = len(self._values)
        print(
            f"Data object with {n} scenarios. Use obj.select(...) to access values dataframe."
        )
        try:
            return self._values._repr_html_()
        except AttributeError:
            print(self._values.__repr__())

    def _check_and_harmonise_inputs(self, other):
        if isinstance(other, (int, float)):
            return self._values, other

        y1, y2 = self._year, other._year

        # Case 1: both are multiple years
        if isinstance(y1, list) and isinstance(y2, list):
            if set(y1) != set(y2):
                raise Exception(f"Years of left var ({y1}) not compatible with ({y2})")
            return self._values, other._values

        # Case 2: self multiple years, other single year
        if isinstance(y1, list) and not isinstance(y2, list):
            # Duplicate values of other to have the same value
            # at every year of self
            harmonised_values2 = pd.DataFrame({year: other._values for year in y1})
            return self._values, harmonised_values2

        # Case 3: inverse of case 2
        if not isinstance(y1, list) and isinstance(y2, list):
            harmonised_values1 = pd.DataFrame({year: self._values for year in y2})
            return harmonised_values1, other._values

        # Case 4: var1 and var2 are single year
        return self._values, other._values

    def __add__(self, other):
        self_values, other_values = Var._check_and_harmonise_inputs(self, other)
        new_values = self_values + other_values
        return Var(
            self.data,
            self.scenarios,
            self.vetted_scenarios,
            values=new_values,
            default=self.default,
        )

    def __sub__(self, other):
        self_values, other_values = Var._check_and_harmonise_inputs(self, other)
        new_values = self_values - other_values
        return Var(
            self.data,
            self.scenarios,
            self.vetted_scenarios,
            values=new_values,
            default=self.default,
        )

    def __mul__(self, other):
        self_values, other_values = Var._check_and_harmonise_inputs(self, other)
        new_values = self_values * other_values
        return Var(
            self.data,
            self.scenarios,
            self.vetted_scenarios,
            values=new_values,
            default=self.default,
        )

    def __truediv__(self, other):
        self_values, other_values = Var._check_and_harmonise_inputs(self, other)
        new_values = self_values / other_values
        return Var(
            self.data,
            self.scenarios,
            self.vetted_scenarios,
            values=new_values,
            default=self.default,
        )


class DataVar:
    def __init__(self, data, scenarios, vetted_scenarios):
        self.data = data
        self.scenarios = scenarios
        self.vetted_scenarios = vetted_scenarios

    def __call__(self, variable, year=None, **kwargs):
        return Var(
            self.data, self.scenarios, self.vetted_scenarios, variable, year, **kwargs
        )


def _to_list(value, to_str=False):
    if isinstance(value, (tuple, list)):
        return [str(v) for v in value] if to_str else value
    return [str(value)] if to_str else [value]
