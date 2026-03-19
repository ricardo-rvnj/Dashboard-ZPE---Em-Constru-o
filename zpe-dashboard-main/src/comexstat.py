import polars as pl
import requests, os
from typing import Dict, List, Optional, Any


class Comexstat:
    """
    Cliente para a API do ComexStat, otimizado para extração de dados.
    """

    def __init__(self):
        self.BASE_URL: str = "https://api-comexstat.mdic.gov.br"
        # Define o caminho para o arquivo de certificado.
        # É uma boa prática usar caminhos relativos ou variáveis de ambiente.
        self.CERT_PATH: str = os.path.join(
            # os.path.dirname(__file__),
            "resources",
            "certificate",
            "mdic-gov-br.pem",
        )

    def _make_request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        json_body: Optional[Dict[str, Any]] = None,
    ) -> Optional[requests.Response]:
        """
        Método utilitário privado para fazer requisições e tratar exceções.
        """
        url = f"{self.BASE_URL}{endpoint}"
        try:
            if method.upper() == "GET":
                response = requests.get(url, params=params, verify=self.CERT_PATH)
            elif method.upper() == "POST":
                response = requests.post(
                    url, json=json_body, params=params, verify=self.CERT_PATH
                )
            else:
                raise ValueError("Método HTTP não suportado.")
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            print(f"Erro na requisição para '{url}': {e}")
            return None

    def get_last_updated_date(
        self, data_type: str = "general"
    ) -> Optional[Dict[str, Any]]:
        endpoint = f"/{data_type}/dates/updated"
        response = self._make_request("GET", endpoint)
        return response.json() if response else None

    def get_available_years(self, data_type: str = "general") -> Optional[List[int]]:
        endpoint = f"/{data_type}/dates/years"
        response = self._make_request("GET", endpoint)
        return response.json() if response else None

    def get_available_filters(
        self, data_type: str = "general", language: str = "pt"
    ) -> Optional[pl.DataFrame]:
        endpoint = f"/{data_type}/filters"
        params = {"language": language}
        response = self._make_request("GET", endpoint, params=params)
        return pl.DataFrame(response.json()["data"]["list"]) if response else None

    def get_filter_values(
        self, filter_name: str, data_type: str = "general", language: str = "pt"
    ) -> Optional[pl.DataFrame]:
        endpoint = f"/{data_type}/filters/{filter_name}"
        params = {"language": language}
        response = self._make_request("GET", endpoint, params=params)
        return pl.DataFrame(response.json()["data"][0]) if response else None

    def get_available_details(
        self, data_type: str = "general", language: str = "pt"
    ) -> Optional[pl.DataFrame]:
        endpoint = f"/{data_type}/details"
        params = {"language": language}
        response = self._make_request("GET", endpoint, params=params)
        return pl.DataFrame(response.json()["data"]["list"]) if response else None

    def get_available_metrics(
        self, data_type: str = "general", language: str = "pt"
    ) -> Optional[pl.DataFrame]:
        endpoint = f"/{data_type}/metrics"
        params = {"language": language}
        response = self._make_request("GET", endpoint, params=params)
        return pl.DataFrame(response.json()["data"]["list"]) if response else None

    def query_comexstat_data(
        self,
        flow: str,
        period_from: str,
        period_to: str,
        data_type: str = "general",
        filters: Optional[List[Dict[str, Any]]] = None,
        details: Optional[List[str]] = None,
        metrics: Optional[List[str]] = None,
        language: str = "pt",
        month_detail: bool = False,
    ) -> Optional[pl.DataFrame]:
        """
        Realiza uma consulta de dados gerais ou por município de exportação ou importação.
        flow: 'export' ou 'import'.
        period_from: data inicial no formato 'YYYY-MM'.
        period_to: data final no formato 'YYYY-MM'.
        data_type: 'general' ou 'cities'.
        filters (opcional): lista de dicionários, e.g., [{'filter': 'state', 'values': [23]}].
        details (opcional): lista de strings, e.g., ['city'].
        metrics (opcional): lista de strings, e.g., ['metricFOB'].
        month_detail: Habilita detalhamento mensal.
        """
        endpoint = f"/{data_type}"
        body = {
            "flow": flow,
            "monthDetail": month_detail,
            "period": {"from": period_from, "to": period_to},
            "filters": filters or [],
            "details": details or [],
            "metrics": metrics or [],
        }
        params = {"language": language}
        response = self._make_request("POST", endpoint, json_body=body, params=params)

        if response and response.json().get("data"):
            # A chave 'list' existe apenas para 'cities' e 'general', mas não para tabelas auxiliares
            data = response.json()["data"].get("list", response.json()["data"])
            return pl.DataFrame(data)

        return pl.DataFrame()

    def get_auxiliary_table(
        self,
        table_name: str,
        language: str = "pt",
        page: int = 1,
        per_page: int = 100,
        add: Optional[str] = None,
        search: Optional[str] = None,
    ) -> Optional[pl.DataFrame]:
        endpoint = f"/tabelas-auxiliares/{table_name}"
        params = {"language": language, "page": page, "perPage": per_page}
        if add:
            params["add"] = add
        if search:
            params["search"] = search

        response = self._make_request("GET", endpoint, params=params)
        return pl.DataFrame(response.json()["data"]) if response else None

    def fetch_comexstat_by_city(
        self, year: int, state_code: int
    ) -> Optional[pl.DataFrame]:
        """
        Extrai dados de exportação do ComexStat por município para um estado e ano.
        """
        return self.query_comexstat_data(
            flow="export",
            period_from=f"{year}-01",
            period_to=f"{year}-12",
            data_type="cities",
            filters=[{"filter": "state", "values": [state_code]}],
            details=["city"],
            metrics=["metricFOB", "metricKG", "metricCIF"],
        )


# Para uso do código:
# api = ComexStat()
# df_ceara_2024 = api.fetch_comexstat_by_city(year=2024, state_code=23)
# if df_ceara_2024 is not None:
#     print(df_ceara_2024)
