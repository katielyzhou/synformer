import pathlib

import click
from synformer.sampler.analog.parallel import run_parallel_sampling, run_sampling_one_cpu

from synformer.chem.mol import Molecule, read_mol_file


def _input_mols_option(p):
    return list(read_mol_file(p))


@click.command()
@click.option("--input", "-i", type=_input_mols_option, required=True)
@click.option("--output", "-o", type=click.Path(exists=False, path_type=pathlib.Path), required=True)
@click.option(
    "--model-path",
    "-m",
    type=click.Path(exists=True, path_type=pathlib.Path),
    default="data/trained_weights/default.ckpt",
)
@click.option("--search-width", type=int, default=24)
@click.option("--exhaustiveness", type=int, default=64)
@click.option("--num-gpus", type=int, default=-1)
@click.option("--num-workers-per-gpu", type=int, default=1)
@click.option("--task-qsize", type=int, default=0)
@click.option("--result-qsize", type=int, default=0)
@click.option("--time-limit", type=int, default=180)
@click.option("--dont-sort", is_flag=True)
def main(
    input: list[Molecule],
    output: pathlib.Path,
    model_path: pathlib.Path,
    search_width: int,
    exhaustiveness: int,
    num_gpus: int,
    num_workers_per_gpu: int,
    task_qsize: int,
    result_qsize: int,
    time_limit: int,
    dont_sort: bool,
):
    run_parallel_sampling(
        input=input,
        output=output,
        model_path=model_path,
        search_width=search_width,
        exhaustiveness=exhaustiveness,
        num_gpus=num_gpus,
        num_workers_per_gpu=num_workers_per_gpu,
        task_qsize=task_qsize,
        result_qsize=result_qsize,
        time_limit=time_limit,
        sort_by_scores=not dont_sort,
    )

def _input_mols_option(p):
    return list(read_mol_file(p))[0]

@click.command()
@click.option("--input", "-i", type=_input_mols_option, required=True)
@click.option("--output", "-o", type=click.Path(exists=False, path_type=pathlib.Path), required=True)
@click.option(
    "--model-path",
    "-m",
    type=click.Path(exists=True, path_type=pathlib.Path),
    default="data/trained_weights/default.ckpt",
)
@click.option(
    "--fpi-path",
    "-m",
    type=click.Path(exists=True, path_type=pathlib.Path),
    default="data/fpindex.pkl",
)
@click.option(
    "--mat-path",
    "-m",
    type=click.Path(exists=True, path_type=pathlib.Path),
    default="data/matrix.pkl",
)
@click.option("--search-width", type=int, default=24)
@click.option("--exhaustiveness", type=int, default=64)
@click.option("--time-limit", type=int, default=180)
@click.option("--max_results", type=int, default=100)
@click.option("--max_evolve_steps", type=int, default=12)
@click.option("--dont-sort", is_flag=True)
def main_cpu(
    input: Molecule,
    output: pathlib.Path,
    model_path: pathlib.Path,
    mat_path: pathlib.Path,
    fpi_path: pathlib.Path,
    search_width: int,
    exhaustiveness: int,
    time_limit: int,
    max_results: int,
    max_evolve_steps: int,
    dont_sort: bool,
):
    run_sampling_one_cpu(
        input=input,
        output=output,
        model_path=model_path,
        mat_path=mat_path,
        fpi_path=fpi_path,
        search_width=search_width,
        exhaustiveness=exhaustiveness,
        time_limit=time_limit,
        max_results = max_results,
        max_evolve_steps = max_evolve_steps,
        sort_by_scores=not dont_sort,
    )



if __name__ == "__main__":
    main_cpu()
