import os, io
import requests
import polars as pl  # Alteração aqui para maior clareza
from pyDataverse.api import NativeApi


def _get_api(api_token: str | None):
    if api_token is None:
        api_key = os.getenv("HARVARD_API_KEY")
        if not api_key:
            raise ValueError(
                "No subscription key provided and 'COMTRADE_API_KEY' not found in environment."
            )
        return api_key  # Retorna a chave do ambiente se api_token for None
    return api_token  # Retorna api_token se ele for fornecido


class HarvardDataverse:
    def __init__(
        self,
        base_url: str = "https://dataverse.harvard.edu",
        api_token: str | None = None,
    ):
        self.api_token = _get_api(api_token)
        self.BASE_URL = base_url

    def _download_files(self, doi: str, target_filename: str | None = None):
        """
        Baixa arquivos de um conjunto de dados do Harvard Dataverse.

        Args:
            self.api_token (str): O token da sua API Dataverse.
            doi (str): O identificador DOI do conjunto de dados.
            target_filename (str, opcional): O nome do arquivo específico a ser baixado.
                                            Se None, todos os arquivos serão baixados.
        """

        api = NativeApi(self.BASE_URL, self.api_token)

        download_dir = doi.replace(":", "_").replace("/", "_")
        os.makedirs(download_dir, exist_ok=True)
        print(f"Os arquivos serão salvos em: {download_dir}")

        response = api.get_dataset(doi)
        if response.status_code != 200:
            print(f"Erro ao acessar o dataset: {response.text}")
            return

        dataset_data = response.json().get("data", {})
        files_list = dataset_data.get("latestVersion", {}).get("files", [])

        if not files_list:
            print("Nenhum arquivo encontrado neste dataset.")
            return

        for file_info in files_list:
            file_name = file_info.get("dataFile", {}).get("filename")
            file_id = file_info.get("dataFile", {}).get("id")

            if target_filename and file_name != target_filename:
                continue

            print(f"Baixando {file_name}...")

            try:
                download_url = f"{self.BASE_URL}/api/access/datafile/{file_id}"
                headers = {"X-Dataverse-key": self.api_token}
                file_response = requests.get(download_url, headers=headers, stream=True)
                file_response.raise_for_status()

                with open(os.path.join(download_dir, file_name), "wb") as f:
                    for chunk in file_response.iter_content(chunk_size=8192):
                        f.write(chunk)

                print(f"Download de {file_name} concluído com sucesso.")

                if target_filename:
                    break

            except requests.exceptions.RequestException as e:
                print(f"Erro ao baixar o arquivo {file_name}: {e}")

        print("Processo de download concluído.")

    def import_df(
        self, doi: str, target_filename: str = None, polars_reader_options: dict = None
    ):
        """
        Baixa arquivos de um conjunto de dados do Harvard Dataverse e os retorna como DataFrames.
        ...
        """
        api = NativeApi(self.BASE_URL, self.api_token)
        dataframes = {}

        response = api.get_dataset(doi)
        if response.status_code != 200:
            print(f"Erro ao acessar o dataset: {response.text}")
            return {}

        dataset_data = response.json().get("data", {})
        files_list = dataset_data.get("latestVersion", {}).get("files", [])

        if not files_list:
            print("Nenhum arquivo encontrado neste dataset.")
            return {}

        if polars_reader_options is None:
            polars_reader_options = {}

        for file_info in files_list:
            file_name = file_info.get("dataFile", {}).get("filename")
            file_id = file_info.get("dataFile", {}).get("id")

            if target_filename and file_name != target_filename:
                continue

            if not file_name.endswith(".csv"):
                print(f"Ignorando o arquivo {file_name}: não é um arquivo CSV.")
                continue

            print(f"Processando {file_name}...")

            try:
                download_url = f"{self.BASE_URL}/api/access/datafile/{file_id}"
                headers = {"X-Dataverse-key": self.api_token}
                file_response = requests.get(download_url, headers=headers, stream=True)
                file_response.raise_for_status()

                file_content = io.BytesIO(file_response.content)

                df = pl.read_csv(
                    file_content, **polars_reader_options
                )  # Alteração crucial para Polars
                dataframes[file_name] = df
                print(f"DataFrame para {file_name} criado com sucesso.")

                if target_filename:
                    break

            except requests.exceptions.RequestException as e:
                print(f"Erro ao baixar o arquivo {file_name}: {e}")
            except Exception as e:
                print(f"Erro ao processar o arquivo {file_name} para DataFrame: {e}")

        if target_filename is not None:
            # Substituição de pd.DataFrame(None) para uma abordagem nativa mais robusta
            return dataframes.get(target_filename, pl.DataFrame())
        return dataframes

    def query_data(self, year: int, doi: str, target_filename: str):
        # Esta função ai    nda precisa ser implementada.
        pass
