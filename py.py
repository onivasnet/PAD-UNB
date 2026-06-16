from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize


APP_TITLE = "Coerencia legislativa na Camara dos Deputados"

THEME_SEEDS = {
	"Seguranca publica": [
		"seguranca", "policia", "crime", "criminal", "penal", "arma", "violencia",
		"prisao", "trafico", "homicidio", "militar", "bombeiro", "guarda",
	],
	"Economia": [
		"economia", "tributo", "imposto", "fiscal", "credito", "orcamento", "empresa",
		"mercado", "renda", "trabalho", "emprego", "salario", "previdencia", "banco",
	],
	"Direitos sociais": [
		"direito", "social", "assistencia", "familia", "mulher", "crianca", "idoso",
		"igualdade", "moradia", "pessoa com deficiencia", "beneficio", "vulneravel",
	],
	"Educacao": [
		"educacao", "ensino", "escola", "universidade", "professor", "aluno", "creche",
		"instituto federal", "bolsa", "formacao", "pesquisa", "tecnologica",
	],
	"Meio ambiente": [
		"meio ambiente", "ambiental", "floresta", "amazonia", "clima", "sustentavel",
		"licenciamento", "desmatamento", "recursos hidricos", "residuo", "energia limpa",
	],
	"Saude": [
		"saude", "sus", "hospital", "medico", "medicamento", "vacina", "doenca",
		"paciente", "tratamento", "enfermagem", "saude mental", "vigilancia sanitaria",
	],
	"Administracao publica": [
		"administracao publica", "servidor", "cargo", "ministerio", "autarquia", "gestao",
		"licitacao", "contrato", "transparencia", "controle", "servico publico", "agencia",
	],
}

VOTE_MAP = {
	"sim": 1.0,
	"nao": -1.0,
	"não": -1.0,
	"abstencao": 0.0,
	"abstenção": 0.0,
	"obstrucao": 0.0,
	"obstrução": 0.0,
	"art. 17": 0.0,
	"branco": 0.0,
}


def app_dir() -> Path:
	return Path(__file__).resolve().parent


def default_data_dir() -> Path:
	return app_dir()


def normalize_key(value: object) -> str:
	text = "" if pd.isna(value) else str(value)
	text = text.strip().lower()
	replacements = str.maketrans("áàãâéêíóôõúç", "aaaaeeiooouc")
	return text.translate(replacements)


def extract_last_number(value: object) -> str | None:
	if pd.isna(value):
		return None
	matches = re.findall(r"\d+", str(value))
	return matches[-1] if matches else None


def find_files(folder: Path, prefixes: Iterable[str]) -> list[Path]:
	files: list[Path] = []
	for prefix in prefixes:
		for extension in ("csv", "json", "xlsx", "xls"):
			files.extend(sorted(folder.glob(f"{prefix}*.{extension}")))
	return prefer_fast_tables([path for path in files if not path.name.endswith(".crdownload") and not path.name.startswith("~$")])


def prefer_fast_tables(paths: list[Path]) -> list[Path]:
	fast_suffixes = {".csv", ".xlsx", ".xls"}
	fast_stems = {path.stem for path in paths if path.suffix.lower() in fast_suffixes}
	return [path for path in paths if path.suffix.lower() != ".json" or path.stem not in fast_stems]


def file_year(path: Path) -> str | None:
	match = re.search(r"(?:^|[-_])(20\d{2})(?:\D|$)", path.stem)
	return match.group(1) if match else None


def filter_files_by_year(paths: list[Path], selected_years: list[str]) -> list[Path]:
	if not selected_years:
		return paths
	return [path for path in paths if file_year(path) is None or file_year(path) in selected_years]


def read_table(path: Path) -> pd.DataFrame:
	if path.suffix.lower() == ".csv":
		try:
			return pd.read_csv(path, sep=";", dtype=str, encoding="utf-8-sig", low_memory=False)
		except Exception:
			return pd.read_csv(path, sep=None, engine="python", dtype=str, encoding="utf-8-sig")
	if path.suffix.lower() == ".json":
		data = pd.read_json(path, dtype=False)
		return pd.json_normalize(data.to_dict(orient="records")).astype(str)
	if path.suffix.lower() in {".xlsx", ".xls"}:
		return pd.read_excel(path, dtype=str)
	return pd.DataFrame()


def join_text_columns(raw: pd.DataFrame, columns: list[str]) -> pd.Series:
	if not columns:
		return pd.Series("", index=raw.index)
	text = raw[columns[0]].fillna("").astype(str)
	for column in columns[1:]:
		text = text.str.cat(raw[column].fillna("").astype(str), sep=" ")
	return text


@st.cache_data(show_spinner=False)
def load_many(file_names: tuple[str, ...]) -> pd.DataFrame:
	frames = []
	for name in file_names:
		path = Path(name)
		try:
			frame = read_table(path)
		except Exception as exc:  # pragma: no cover - displayed in the UI
			st.warning(f"Nao foi possivel ler {path.name}: {exc}")
			continue
		if not frame.empty:
			frame["_arquivo_origem"] = path.name
			frames.append(frame)
	return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def choose_column(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
	normalized = {normalize_key(column): column for column in df.columns}
	for candidate in candidates:
		key = normalize_key(candidate)
		if key in normalized:
			return normalized[key]
	return None


def prepare_deputies(raw: pd.DataFrame) -> pd.DataFrame:
	if raw.empty:
		return pd.DataFrame(columns=["deputado_id", "deputado_nome"])
	id_column = choose_column(raw, ["id", "deputado_id", "idDeputado", "uri"])
	name_column = choose_column(raw, ["nome", "deputado_nome", "nomeCivil"])
	party_column = choose_column(raw, ["siglaPartido", "deputado_siglaPartido", "partido"])
	uf_column = choose_column(raw, ["siglaUf", "uf", "deputado_siglaUf"])

	deputies = pd.DataFrame()
	deputies["deputado_id"] = raw[id_column].map(extract_last_number) if id_column else pd.NA
	deputies["deputado_nome"] = raw[name_column] if name_column else pd.NA
	deputies["partido"] = raw[party_column] if party_column else pd.NA
	deputies["uf"] = raw[uf_column] if uf_column else pd.NA
	deputies = deputies.dropna(subset=["deputado_id", "deputado_nome"]).drop_duplicates("deputado_id")
	return deputies


def prepare_propositions(raw: pd.DataFrame) -> pd.DataFrame:
	if raw.empty:
		return pd.DataFrame(columns=["proposicao_id", "texto_proposicao", "rotulo_proposicao"])
	id_column = choose_column(raw, ["id", "idProposicao", "proposicao_id", "uri"])
	type_column = choose_column(raw, ["siglaTipo", "proposicao_siglaTipo"])
	number_column = choose_column(raw, ["numero", "proposicao_numero"])
	year_column = choose_column(raw, ["ano", "proposicao_ano"])
	text_columns = [
		column for column in [
			choose_column(raw, ["ementa", "proposicao_ementa"]),
			choose_column(raw, ["ementaDetalhada"]),
			choose_column(raw, ["keywords"]),
		]
		if column
	]

	propositions = pd.DataFrame()
	propositions["proposicao_id"] = raw[id_column].map(extract_last_number) if id_column else pd.NA
	propositions["texto_proposicao"] = join_text_columns(raw, text_columns)
	propositions["rotulo_proposicao"] = (
		raw.get(type_column, "").fillna("").astype(str)
		+ " "
		+ raw.get(number_column, "").fillna("").astype(str)
		+ "/"
		+ raw.get(year_column, "").fillna("").astype(str)
	).str.strip()
	return propositions.dropna(subset=["proposicao_id"]).drop_duplicates("proposicao_id")


def prepare_authors(raw: pd.DataFrame) -> pd.DataFrame:
	if raw.empty:
		return pd.DataFrame(columns=["proposicao_id", "deputado_id", "deputado_nome", "partido", "uf", "coautoria"])
	prop_column = choose_column(raw, ["idProposicao", "proposicao_id", "uriProposicao"])
	deputy_column = choose_column(raw, ["idDeputadoAutor", "deputado_id", "uriAutor"])
	name_column = choose_column(raw, ["nomeAutor", "deputado_nome"])
	party_column = choose_column(raw, ["siglaPartidoAutor", "deputado_siglaPartido", "partido"])
	uf_column = choose_column(raw, ["siglaUFAutor", "deputado_siglaUf", "uf"])
	order_column = choose_column(raw, ["ordemAssinatura"])
	type_column = choose_column(raw, ["tipoAutor"])

	authors = pd.DataFrame()
	authors["proposicao_id"] = raw[prop_column].map(extract_last_number) if prop_column else pd.NA
	authors["deputado_id"] = raw[deputy_column].map(extract_last_number) if deputy_column else pd.NA
	authors["deputado_nome"] = raw[name_column] if name_column else pd.NA
	authors["partido"] = raw[party_column] if party_column else pd.NA
	authors["uf"] = raw[uf_column] if uf_column else pd.NA
	authors["coautoria"] = pd.to_numeric(raw[order_column], errors="coerce").fillna(1).gt(1) if order_column else False
	if type_column:
		authors = authors[raw[type_column].fillna("").str.contains("Deputado", case=False, na=False)]
	return authors.dropna(subset=["proposicao_id", "deputado_id"]).drop_duplicates()


def classify_themes(propositions: pd.DataFrame) -> pd.DataFrame:
	if propositions.empty:
		return propositions.assign(tema="Sem classificacao", confianca_tema=0.0)

	texts = propositions["texto_proposicao"].fillna("").astype(str)
	if texts.str.strip().eq("").all():
		return propositions.assign(tema="Sem classificacao", confianca_tema=0.0)

	seed_vocabulary = sorted({normalize_key(seed) for seeds in THEME_SEEDS.values() for seed in seeds})
	vectorizer = TfidfVectorizer(
		lowercase=True,
		strip_accents="unicode",
		ngram_range=(1, 3),
		vocabulary=seed_vocabulary,
	)
	matrix = vectorizer.fit_transform(texts)
	feature_names = np.array(vectorizer.get_feature_names_out())

	theme_scores = np.zeros((len(propositions), len(THEME_SEEDS)))
	for theme_index, (_, seeds) in enumerate(THEME_SEEDS.items()):
		seed_keys = {normalize_key(seed) for seed in seeds}
		feature_mask = np.array([normalize_key(feature) in seed_keys for feature in feature_names])
		if feature_mask.any():
			theme_scores[:, theme_index] = np.asarray(matrix[:, feature_mask].sum(axis=1)).ravel()

	best = theme_scores.argmax(axis=1)
	confidence = theme_scores.max(axis=1)
	themes = np.array(list(THEME_SEEDS.keys()))[best]
	themes = np.where(confidence > 0, themes, "Sem classificacao")

	classified = propositions.copy()
	classified["tema"] = themes
	classified["confianca_tema"] = confidence
	return classified


def prepare_votes(raw_votes: pd.DataFrame, raw_vote_objects: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
	if raw_votes.empty:
		return pd.DataFrame(), pd.DataFrame()
	vote_column = choose_column(raw_votes, ["voto"])
	voting_column = choose_column(raw_votes, ["idVotacao"])
	deputy_column = choose_column(raw_votes, ["deputado_id"])
	name_column = choose_column(raw_votes, ["deputado_nome"])
	party_column = choose_column(raw_votes, ["deputado_siglaPartido"])
	uf_column = choose_column(raw_votes, ["deputado_siglaUf"])
	if not all([vote_column, voting_column, deputy_column]):
		return pd.DataFrame(), pd.DataFrame()

	votes = pd.DataFrame()
	votes["idVotacao"] = raw_votes[voting_column]
	votes["deputado_id"] = raw_votes[deputy_column].map(extract_last_number)
	votes["deputado_nome"] = raw_votes[name_column] if name_column else pd.NA
	votes["partido"] = raw_votes[party_column] if party_column else pd.NA
	votes["uf"] = raw_votes[uf_column] if uf_column else pd.NA
	votes["voto"] = raw_votes[vote_column]
	votes["voto_valor"] = raw_votes[vote_column].map(lambda value: VOTE_MAP.get(normalize_key(value), 0.0))
	votes = votes.dropna(subset=["idVotacao", "deputado_id"])

	vote_objects = pd.DataFrame()
	if not raw_vote_objects.empty:
		object_voting_column = choose_column(raw_vote_objects, ["idVotacao"])
		object_prop_column = choose_column(raw_vote_objects, ["proposicao_id", "proposicao_uri"])
		object_text_column = choose_column(raw_vote_objects, ["proposicao_ementa", "descricao"])
		if object_voting_column:
			vote_objects["idVotacao"] = raw_vote_objects[object_voting_column]
			vote_objects["proposicao_id"] = raw_vote_objects[object_prop_column].map(extract_last_number) if object_prop_column else pd.NA
			vote_objects["texto_votacao"] = raw_vote_objects[object_text_column] if object_text_column else ""
			vote_objects = vote_objects.drop_duplicates("idVotacao")
	return votes, vote_objects


def prepare_groups(raw: pd.DataFrame) -> pd.DataFrame:
	if raw.empty:
		return pd.DataFrame(columns=["deputado_id", "grupo_nome"])
	uri_column = choose_column(raw, ["membro_uri", "deputado_uri"])
	name_column = choose_column(raw, ["membro_nome", "deputado_nome"])
	group_column = choose_column(raw, ["nomeGrupo", "titulo"])
	member_type_column = choose_column(raw, ["membro_tipo"])
	if not group_column:
		return pd.DataFrame(columns=["deputado_id", "grupo_nome"])

	groups = pd.DataFrame()
	groups["deputado_id"] = raw[uri_column].map(extract_last_number) if uri_column else pd.NA
	groups["deputado_nome"] = raw[name_column] if name_column else pd.NA
	groups["grupo_nome"] = raw[group_column]
	if member_type_column:
		groups = groups[raw[member_type_column].fillna("").str.contains("Deputado", case=False, na=False)]
	return groups.dropna(subset=["deputado_id", "grupo_nome"]).drop_duplicates()


def theme_profile_by_deputy(authors: pd.DataFrame, propositions: pd.DataFrame) -> pd.DataFrame:
	authored = authors.merge(propositions[["proposicao_id", "tema"]], on="proposicao_id", how="left")
	authored["tema"] = authored["tema"].fillna("Sem classificacao")
	counts = pd.crosstab(authored["deputado_id"], authored["tema"])
	counts = counts.reindex(columns=list(THEME_SEEDS.keys()) + ["Sem classificacao"], fill_value=0)
	return counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0).add_prefix("tema_")


def vote_profile_by_deputy(votes: pd.DataFrame, max_votes: int) -> pd.DataFrame:
	if votes.empty:
		return pd.DataFrame()
	relevant_votes = votes["idVotacao"].value_counts().head(max_votes).index
	pivot = votes[votes["idVotacao"].isin(relevant_votes)].pivot_table(
		index="deputado_id",
		columns="idVotacao",
		values="voto_valor",
		aggfunc="mean",
		fill_value=0,
	)
	return pivot.add_prefix("voto_")


def group_profile_by_deputy(groups: pd.DataFrame, max_groups: int) -> pd.DataFrame:
	if groups.empty:
		return pd.DataFrame()
	relevant_groups = groups["grupo_nome"].value_counts().head(max_groups).index
	selected = groups[groups["grupo_nome"].isin(relevant_groups)].copy()
	selected["presenca"] = 1
	pivot = selected.pivot_table(
		index="deputado_id",
		columns="grupo_nome",
		values="presenca",
		aggfunc="max",
		fill_value=0,
	)
	return pivot.add_prefix("grupo_")


def combine_profiles(
	base_people: pd.DataFrame,
	theme_profile: pd.DataFrame,
	vote_profile: pd.DataFrame,
	group_profile: pd.DataFrame,
	text_weight: float,
	vote_weight: float,
	group_weight: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
	profile_parts = []
	if not theme_profile.empty:
		profile_parts.append(theme_profile * text_weight)
	if not vote_profile.empty:
		profile_parts.append(vote_profile * vote_weight)
	if not group_profile.empty:
		profile_parts.append(group_profile * group_weight)

	if profile_parts:
		features = pd.concat(profile_parts, axis=1).fillna(0)
	else:
		features = pd.DataFrame(index=base_people["deputado_id"].astype(str).unique())

	people = base_people.copy()
	people["deputado_id"] = people["deputado_id"].astype(str)
	features.index = features.index.astype(str)
	people = people.drop_duplicates("deputado_id").set_index("deputado_id")
	people = people.join(features, how="inner")
	feature_columns = [column for column in people.columns if column.startswith(("tema_", "voto_", "grupo_"))]
	return people.reset_index(), people[feature_columns].fillna(0)


def compute_coherence(people: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
	if people.empty or features.empty or "partido" not in people:
		return people.assign(coerencia=np.nan)

	people = people.copy().set_index("deputado_id")
	features = features.copy()
	features.index = people.index
	party_vectors = features.groupby(people["partido"].fillna("Sem partido")).mean()

	coherences = []
	for deputy_id, row in features.iterrows():
		party = people.loc[deputy_id, "partido"] if pd.notna(people.loc[deputy_id, "partido"]) else "Sem partido"
		party_vector = party_vectors.loc[party].to_numpy().reshape(1, -1)
		deputy_vector = row.to_numpy().reshape(1, -1)
		if np.linalg.norm(deputy_vector) == 0 or np.linalg.norm(party_vector) == 0:
			coherences.append(np.nan)
		else:
			coherences.append(float(cosine_similarity(deputy_vector, party_vector)[0, 0]))
	people["coerencia"] = coherences
	return people.reset_index()


def add_projection_and_clusters(people: pd.DataFrame, features: pd.DataFrame, cluster_count: int) -> pd.DataFrame:
	result = people.copy()
	if len(result) < 2 or features.empty:
		result["x"] = 0.0
		result["y"] = 0.0
		result["cluster"] = "Sem cluster"
		return result

	matrix = normalize(features.to_numpy(), norm="l2")
	components = PCA(n_components=2, random_state=42).fit_transform(matrix)
	result["x"] = components[:, 0]
	result["y"] = components[:, 1]

	cluster_count = max(2, min(cluster_count, len(result) - 1))
	labels = KMeans(n_clusters=cluster_count, random_state=42, n_init="auto").fit_predict(matrix)
	result["cluster"] = pd.Series(labels, index=result.index).astype(str)
	return result


def build_base_people(deputies: pd.DataFrame, authors: pd.DataFrame, votes: pd.DataFrame) -> pd.DataFrame:
	frames = []
	for frame in [deputies, authors, votes]:
		available = [column for column in ["deputado_id", "deputado_nome", "partido", "uf"] if column in frame]
		if available:
			frames.append(frame[available])
	if not frames:
		return pd.DataFrame(columns=["deputado_id", "deputado_nome", "partido", "uf"])
	people = pd.concat(frames, ignore_index=True, sort=False)
	people = people.dropna(subset=["deputado_id"]).sort_values(["deputado_id", "partido"], na_position="last")
	people = people.groupby("deputado_id", as_index=False).first()
	return people


def build_party_theme_matrix(people: pd.DataFrame, topic_columns: list[str]) -> pd.DataFrame:
	if not topic_columns or people.empty:
		return pd.DataFrame()
	matrix = people.groupby("partido", dropna=False)[topic_columns].mean().reset_index()
	matrix["partido"] = matrix["partido"].fillna("Sem partido")
	matrix = matrix.rename(columns={column: column.replace("tema_", "") for column in topic_columns})
	return matrix


def build_theme_long(people: pd.DataFrame, topic_columns: list[str]) -> pd.DataFrame:
	if not topic_columns or people.empty:
		return pd.DataFrame(columns=["deputado_id", "deputado_nome", "partido", "tema", "peso"])
	long = people[["deputado_id", "deputado_nome", "partido"] + topic_columns].melt(
		id_vars=["deputado_id", "deputado_nome", "partido"],
		value_vars=topic_columns,
		var_name="tema",
		value_name="peso",
	)
	long["tema"] = long["tema"].str.replace("tema_", "", regex=False)
	long["partido"] = long["partido"].fillna("Sem partido")
	return long


def build_cluster_party_matrix(people: pd.DataFrame) -> pd.DataFrame:
	if people.empty or "cluster" not in people:
		return pd.DataFrame()
	matrix = pd.crosstab(people["cluster"], people["partido"].fillna("Sem partido"))
	return matrix.reset_index()


def build_vote_alignment(votes: pd.DataFrame) -> pd.DataFrame:
	if votes.empty:
		return pd.DataFrame(columns=["deputado_id", "alinhamento_votos_partido"])
	party_vote = votes.groupby(["partido", "idVotacao"], dropna=False)["voto_valor"].mean().rename("media_partido")
	alignment = votes.join(party_vote, on=["partido", "idVotacao"])
	alignment["alinhado"] = np.where(
		alignment["media_partido"].abs().lt(0.01),
		0.5,
		(np.sign(alignment["voto_valor"]) == np.sign(alignment["media_partido"])).astype(float),
	)
	return alignment.groupby("deputado_id", as_index=False)["alinhado"].mean().rename(
		columns={"alinhado": "alinhamento_votos_partido"}
	)


def normalized_crosstab(rows: pd.Series, columns: pd.Series) -> pd.DataFrame:
	if rows.empty or columns.empty:
		return pd.DataFrame()
	table = pd.crosstab(rows.fillna("Sem informacao"), columns.fillna("Sem informacao"))
	return table.div(table.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)


def melt_feature_means(frame: pd.DataFrame, group_column: str, feature_columns: list[str], value_name: str) -> pd.DataFrame:
	if frame.empty or not feature_columns or group_column not in frame:
		return pd.DataFrame(columns=[group_column, "variavel", value_name])
	means = frame.groupby(group_column, dropna=False)[feature_columns].mean().reset_index()
	melted = means.melt(id_vars=group_column, var_name="variavel", value_name=value_name)
	melted["variavel"] = melted["variavel"].str.replace("tema_", "", regex=False)
	return melted


def render_data_sources(folder: Path) -> dict[str, list[Path]]:
	st.sidebar.header("Dados")
	folder_text = st.sidebar.text_input("Pasta dos arquivos", value=str(folder))
	selected_folder = Path(folder_text).expanduser()
	exact_propositions = [selected_folder / "proposicoes.csv"] if (selected_folder / "proposicoes.csv").exists() else []
	sources = {
		"deputados": find_files(selected_folder, ["deputados"]),
		"proposicoes": find_files(selected_folder, ["proposicoes-"])
		+ find_files(selected_folder, ["proposicoes_"])
		+ exact_propositions,
		"autores": find_files(selected_folder, ["proposicoesAutores"]),
		"votos": find_files(selected_folder, ["votacoesVotos"]),
		"objetos_votacao": find_files(selected_folder, ["votacoesObjetos"]),
		"frentes_grupos": find_files(selected_folder, ["frentesDeputados", "gruposMembros"]),
	}

	available_years = sorted({year for paths in sources.values() for path in paths if (year := file_year(path))})
	default_years = available_years[-1:] if available_years else []
	selected_years = st.sidebar.multiselect("Ano dos arquivos", available_years, default=default_years)
	sources = {label: filter_files_by_year(paths, selected_years) for label, paths in sources.items()}

	for label, paths in sources.items():
		st.sidebar.caption(f"{label}: {len(paths)} arquivo(s)")
	return sources


def main() -> None:
	st.set_page_config(page_title=APP_TITLE, layout="wide")
	st.title(APP_TITLE)

	sources = render_data_sources(default_data_dir())
	st.sidebar.header("Modelo")
	text_weight = st.sidebar.slider("Peso dos projetos", 0.0, 3.0, 1.5, 0.1)
	vote_weight = st.sidebar.slider("Peso das votacoes", 0.0, 3.0, 1.0, 0.1)
	group_weight = st.sidebar.slider("Peso de frentes/grupos", 0.0, 3.0, 0.6, 0.1)
	max_votes = st.sidebar.slider("Votacoes nominais no vetor", 10, 500, 150, 10)
	max_groups = st.sidebar.slider("Frentes/grupos no vetor", 5, 200, 60, 5)
	cluster_count = st.sidebar.slider("Numero de clusters", 2, 15, 6, 1)

	load_status = st.empty()
	with st.spinner("Carregando e analisando os arquivos selecionados..."):
		load_status.info("Lendo arquivos da pasta selecionada...")
		raw_deputies = load_many(tuple(str(path) for path in sources["deputados"]))
		raw_propositions = load_many(tuple(str(path) for path in sources["proposicoes"]))
		raw_authors = load_many(tuple(str(path) for path in sources["autores"]))
		raw_votes = load_many(tuple(str(path) for path in sources["votos"]))
		raw_vote_objects = load_many(tuple(str(path) for path in sources["objetos_votacao"]))
		raw_groups = load_many(tuple(str(path) for path in sources["frentes_grupos"]))

		load_status.info("Preparando deputados, proposicoes, votos e frentes...")
		deputies = prepare_deputies(raw_deputies)
		propositions = classify_themes(prepare_propositions(raw_propositions))
		authors = prepare_authors(raw_authors)
		votes, vote_objects = prepare_votes(raw_votes, raw_vote_objects)
		groups = prepare_groups(raw_groups)

		load_status.info("Construindo vetores de deputados e partidos...")
		base_people = build_base_people(deputies, authors, votes)
		theme_profile = theme_profile_by_deputy(authors, propositions)
		vote_profile = vote_profile_by_deputy(votes, max_votes)
		group_profile = group_profile_by_deputy(groups, max_groups)
		people, features = combine_profiles(
			base_people,
			theme_profile,
			vote_profile,
			group_profile,
			text_weight,
			vote_weight,
			group_weight,
		)
		load_status.info("Calculando coerencia, PCA e clusters...")
		people = compute_coherence(people, features)
		people = add_projection_and_clusters(people, features, cluster_count)
	load_status.empty()

	if people.empty:
		st.error("Nao encontrei deputados com dados suficientes. Confira a pasta e os arquivos carregados.")
		return

	topic_columns = [column for column in people.columns if column.startswith("tema_")]
	people["tema_dominante"] = (
		people[topic_columns].idxmax(axis=1).str.replace("tema_", "", regex=False)
		if topic_columns else "Sem classificacao"
	)

	coherence_mean = people["coerencia"].mean(skipna=True)
	party_summary = people.groupby("partido", dropna=False).agg(
		deputados=("deputado_id", "count"),
		coerencia_media=("coerencia", "mean"),
		coerencia_mediana=("coerencia", "median"),
		coerencia_minima=("coerencia", "min"),
		coerencia_maxima=("coerencia", "max"),
		heterogeneidade=("coerencia", lambda values: 1 - values.mean(skipna=True)),
	).reset_index().sort_values("deputados", ascending=False)
	party_theme = melt_feature_means(people, "partido", topic_columns, "peso_medio")
	cluster_theme = melt_feature_means(people, "cluster", topic_columns, "peso_medio")
	cluster_party = normalized_crosstab(people["cluster"], people["partido"])
	party_cluster = normalized_crosstab(people["partido"], people["cluster"])
	coauthor_summary = authors.groupby("deputado_id", as_index=False).agg(
		projetos=("proposicao_id", "nunique"),
		coautorias=("coautoria", "sum"),
	)
	people = people.merge(coauthor_summary, on="deputado_id", how="left")
	people[["projetos", "coautorias"]] = people[["projetos", "coautorias"]].fillna(0)
	if not votes.empty:
		vote_participation = votes.groupby("deputado_id", as_index=False).agg(
			votos_registrados=("idVotacao", "count"),
			posicao_media_voto=("voto_valor", "mean"),
		)
		people = people.merge(vote_participation, on="deputado_id", how="left")
	else:
		people["votos_registrados"] = 0
		people["posicao_media_voto"] = 0.0
	people[["votos_registrados", "posicao_media_voto"]] = people[["votos_registrados", "posicao_media_voto"]].fillna(0)

	col1, col2, col3, col4 = st.columns(4)
	col1.metric("Deputados analisados", f"{len(people):,.0f}".replace(",", "."))
	col2.metric("Proposicoes classificadas", f"{len(propositions):,.0f}".replace(",", "."))
	col3.metric("Votos nominais", f"{len(votes):,.0f}".replace(",", "."))
	col4.metric("Coerencia media", "n/d" if np.isnan(coherence_mean) else f"{coherence_mean:.3f}")

	st.subheader("Mapa politico por atuacao")
	fig_map = px.scatter(
		people,
		x="x",
		y="y",
		color="partido",
		symbol="cluster",
		hover_name="deputado_nome",
		hover_data={"coerencia": ":.3f", "tema_dominante": True, "uf": True, "x": False, "y": False},
		height=620,
	)
	fig_map.update_layout(xaxis_title="PCA 1", yaxis_title="PCA 2", legend_title="Partido")
	st.plotly_chart(fig_map, width="stretch")

	tab_overview, tab_parties, tab_themes, tab_clusters, tab_votes, tab_hypotheses, tab_data = st.tabs(
		["Deputados", "Partidos", "Temas", "Clusters", "Votacoes", "Hipoteses", "Dados"]
	)

	with tab_overview:
		left, right = st.columns(2)
		with left:
			fig_hist = px.histogram(
				people.dropna(subset=["coerencia"]),
				x="coerencia",
				nbins=25,
				color="tema_dominante",
				height=360,
			)
			fig_hist.update_layout(xaxis_title="Coerencia", yaxis_title="Deputados", legend_title="Tema dominante")
			st.plotly_chart(fig_hist, width="stretch")
		with right:
			top_party_names = party_summary.head(12)["partido"].dropna().tolist()
			fig_box = px.box(
				people[people["partido"].isin(top_party_names)],
				x="partido",
				y="coerencia",
				color="partido",
				height=360,
			)
			fig_box.update_layout(showlegend=False, xaxis_title="Partido", yaxis_title="Coerencia")
			st.plotly_chart(fig_box, width="stretch")

		low_col, high_col = st.columns(2)
		with low_col:
			st.write("Deputados menos alinhados ao vetor medio do partido")
			st.dataframe(
				people[["deputado_nome", "partido", "uf", "coerencia", "projetos", "votos_registrados", "tema_dominante"]]
				.sort_values("coerencia", ascending=True, na_position="last")
				.head(20),
				width="stretch",
				hide_index=True,
			)
		with high_col:
			st.write("Deputados mais alinhados ao vetor medio do partido")
			st.dataframe(
				people[["deputado_nome", "partido", "uf", "coerencia", "projetos", "votos_registrados", "tema_dominante"]]
				.sort_values("coerencia", ascending=False, na_position="last")
				.head(20),
				width="stretch",
				hide_index=True,
			)

		st.dataframe(
			people[["deputado_nome", "partido", "uf", "coerencia", "cluster", "tema_dominante", "projetos", "coautorias", "votos_registrados"]]
			.sort_values("coerencia", ascending=True, na_position="last"),
			width="stretch",
			hide_index=True,
		)
		st.download_button(
			"Baixar base de deputados analisada",
			people.to_csv(index=False, sep=";").encode("utf-8-sig"),
			file_name="dashboard_coerencia_deputados.csv",
			mime="text/csv",
		)

	with tab_parties:
		left, right = st.columns(2)
		with left:
			fig_party = px.bar(
				party_summary.dropna(subset=["partido"]),
				x="partido",
				y="coerencia_media",
				color="deputados",
				hover_data={"heterogeneidade": ":.3f", "coerencia_mediana": ":.3f", "deputados": True},
				height=420,
			)
			fig_party.update_layout(xaxis_title="Partido", yaxis_title="Coerencia media")
			st.plotly_chart(fig_party, width="stretch")
		with right:
			fig_heterogeneity = px.scatter(
				party_summary.dropna(subset=["partido", "coerencia_media"]),
				x="deputados",
				y="heterogeneidade",
				size="deputados",
				color="coerencia_media",
				text="partido",
				height=420,
			)
			fig_heterogeneity.update_layout(xaxis_title="Tamanho da bancada", yaxis_title="Heterogeneidade ideologica")
			st.plotly_chart(fig_heterogeneity, width="stretch")

		if not party_theme.empty:
			selected_parties = party_summary.head(15)["partido"].dropna().tolist()
			heatmap_data = party_theme[party_theme["partido"].isin(selected_parties)].pivot(
				index="partido", columns="variavel", values="peso_medio"
			).fillna(0)
			fig_party_theme = px.imshow(
				heatmap_data,
				text_auto=".2f",
				aspect="auto",
				color_continuous_scale="Tealrose",
				height=480,
			)
			fig_party_theme.update_layout(xaxis_title="Tema", yaxis_title="Partido", coloraxis_colorbar_title="Peso")
			st.plotly_chart(fig_party_theme, width="stretch")
		st.dataframe(party_summary, width="stretch", hide_index=True)

	with tab_themes:
		theme_counts = propositions["tema"].value_counts().reset_index()
		theme_counts.columns = ["tema", "proposicoes"]
		left, right = st.columns(2)
		with left:
			fig_themes = px.bar(theme_counts, x="tema", y="proposicoes", color="tema", height=420)
			fig_themes.update_layout(showlegend=False, xaxis_title="Tema", yaxis_title="Proposicoes")
			st.plotly_chart(fig_themes, width="stretch")
		with right:
			fig_theme_confidence = px.violin(
				propositions,
				x="tema",
				y="confianca_tema",
				color="tema",
				box=True,
				height=420,
			)
			fig_theme_confidence.update_layout(showlegend=False, xaxis_title="Tema", yaxis_title="Confianca TF-IDF")
			st.plotly_chart(fig_theme_confidence, width="stretch")

		if not party_theme.empty:
			selected_parties = party_summary.head(12)["partido"].dropna().tolist()
			fig_theme_party = px.bar(
				party_theme[party_theme["partido"].isin(selected_parties)],
				x="partido",
				y="peso_medio",
				color="variavel",
				barmode="stack",
				height=460,
			)
			fig_theme_party.update_layout(xaxis_title="Partido", yaxis_title="Peso medio no vetor", legend_title="Tema")
			st.plotly_chart(fig_theme_party, width="stretch")

		if not cluster_theme.empty:
			fig_theme_cluster = px.bar(
				cluster_theme,
				x="cluster",
				y="peso_medio",
				color="variavel",
				barmode="stack",
				height=420,
			)
			fig_theme_cluster.update_layout(xaxis_title="Cluster", yaxis_title="Peso medio no vetor", legend_title="Tema")
			st.plotly_chart(fig_theme_cluster, width="stretch")

		st.dataframe(
			propositions[["rotulo_proposicao", "tema", "confianca_tema", "texto_proposicao"]]
			.sort_values("confianca_tema", ascending=False),
			width="stretch",
			hide_index=True,
		)

	with tab_clusters:
		cluster_summary = people.groupby("cluster", as_index=False).agg(
			deputados=("deputado_id", "count"),
			coerencia_media=("coerencia", "mean"),
			partidos=("partido", "nunique"),
			tema_mais_comum=("tema_dominante", lambda values: values.mode().iat[0] if not values.mode().empty else "Sem classificacao"),
		).sort_values("deputados", ascending=False)
		left, right = st.columns(2)
		with left:
			fig_cluster_size = px.bar(
				cluster_summary,
				x="cluster",
				y="deputados",
				color="coerencia_media",
				hover_data={"partidos": True, "tema_mais_comum": True},
				height=380,
			)
			fig_cluster_size.update_layout(xaxis_title="Cluster", yaxis_title="Deputados")
			st.plotly_chart(fig_cluster_size, width="stretch")
		with right:
			fig_cluster_box = px.box(people, x="cluster", y="coerencia", color="cluster", height=380)
			fig_cluster_box.update_layout(showlegend=False, xaxis_title="Cluster", yaxis_title="Coerencia")
			st.plotly_chart(fig_cluster_box, width="stretch")

		if not cluster_party.empty:
			fig_cluster_party = px.imshow(
				cluster_party,
				text_auto=".2f",
				aspect="auto",
				color_continuous_scale="RdBu",
				height=480,
			)
			fig_cluster_party.update_layout(xaxis_title="Partido", yaxis_title="Cluster", coloraxis_colorbar_title="Proporcao")
			st.plotly_chart(fig_cluster_party, width="stretch")

		if not party_cluster.empty:
			fig_party_cluster = px.imshow(
				party_cluster,
				text_auto=".2f",
				aspect="auto",
				color_continuous_scale="PuOr",
				height=480,
			)
			fig_party_cluster.update_layout(xaxis_title="Cluster", yaxis_title="Partido", coloraxis_colorbar_title="Proporcao")
			st.plotly_chart(fig_party_cluster, width="stretch")
		st.dataframe(cluster_summary, width="stretch", hide_index=True)

	with tab_votes:
		if votes.empty:
			st.info("Nao ha arquivo de votos nominais carregado para esta selecao.")
		else:
			vote_by_party = votes.groupby(["partido", "voto"], dropna=False).size().reset_index(name="quantidade")
			party_position = votes.groupby("partido", as_index=False).agg(
				votos=("idVotacao", "count"),
				posicao_media=("voto_valor", "mean"),
				votacoes=("idVotacao", "nunique"),
			).sort_values("votos", ascending=False)
			left, right = st.columns(2)
			with left:
				fig_vote_party = px.bar(
					vote_by_party,
					x="partido",
					y="quantidade",
					color="voto",
					barmode="stack",
					height=420,
				)
				fig_vote_party.update_layout(xaxis_title="Partido", yaxis_title="Votos registrados", legend_title="Voto")
				st.plotly_chart(fig_vote_party, width="stretch")
			with right:
				fig_position = px.scatter(
					party_position,
					x="votos",
					y="posicao_media",
					size="votacoes",
					color="posicao_media",
					text="partido",
					height=420,
				)
				fig_position.update_layout(xaxis_title="Votos registrados", yaxis_title="Posicao media (-1 Nao, +1 Sim)")
				st.plotly_chart(fig_position, width="stretch")

			fig_deputy_vote = px.scatter(
				people,
				x="votos_registrados",
				y="coerencia",
				color="partido",
				size="projetos",
				hover_name="deputado_nome",
				height=460,
			)
			fig_deputy_vote.update_layout(xaxis_title="Votos registrados", yaxis_title="Coerencia com o partido")
			st.plotly_chart(fig_deputy_vote, width="stretch")
			st.dataframe(party_position, width="stretch", hide_index=True)

	with tab_hypotheses:
		h1_threshold = st.slider("Limiar de alta coerencia para H1", 0.50, 0.95, 0.75, 0.01)
		h1_share = people["coerencia"].ge(h1_threshold).mean()
		st.metric("H1: proporcao acima do limiar", f"{h1_share:.1%}")

		valid_party_summary = party_summary.dropna(subset=["coerencia_media"]).copy()
		if len(valid_party_summary) > 2:
			correlation = valid_party_summary["deputados"].corr(valid_party_summary["heterogeneidade"])
			st.metric("H2: correlacao tamanho x heterogeneidade", f"{correlation:.3f}")
			st.plotly_chart(
				px.scatter(
					valid_party_summary,
					x="deputados",
					y="heterogeneidade",
					text="partido",
					height=420,
				),
				width="stretch",
			)

		if people["partido"].nunique(dropna=True) > 1 and people["cluster"].nunique() > 1:
			party_codes = people["partido"].fillna("Sem partido").astype("category").cat.codes
			cluster_codes = people["cluster"].astype("category").cat.codes
			ari = adjusted_rand_score(party_codes, cluster_codes)
			st.metric("H3: coincidencia clusters x partidos (ARI)", f"{ari:.3f}")
			st.caption("ARI perto de 1 indica forte coincidencia; perto de 0 sugere agrupamentos pouco alinhados aos partidos formais.")
		if len(people) > cluster_count and people["cluster"].nunique() > 1:
			st.metric("Qualidade interna dos clusters (silhueta)", f"{silhouette_score(normalize(features.to_numpy()), people['cluster']):.3f}")

	with tab_data:
		st.write("Arquivos carregados")
		source_rows = [
			{"tipo": label, "arquivo": path.name, "caminho": str(path)}
			for label, paths in sources.items()
			for path in paths
		]
		st.dataframe(pd.DataFrame(source_rows), width="stretch", hide_index=True)
		st.write("Metodologia operacional")
		st.markdown(
			"""
			- Os projetos sao classificados por TF-IDF a partir de ementa, ementa detalhada, palavras-chave e despacho quando presentes.
			- O vetor do deputado combina temas de proposicoes, comportamento em votacoes nominais e participacao em frentes ou grupos.
			- O vetor do partido e calculado pelos padroes medios dos seus deputados; nao ha classificacao ideologica manual de partidos.
			- A coerencia e a similaridade do cosseno entre o vetor do deputado e o vetor medio de seu partido.
			- O mapa politico usa PCA e os grupos reais sao estimados por K-Means.
			"""
		)
		if not vote_objects.empty:
			st.write("Objetos de votacao identificados")
			st.dataframe(vote_objects, width="stretch", hide_index=True)


if __name__ == "__main__":
	main()
