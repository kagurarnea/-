"""Complete finite supplementary experiment; main model outputs are retained."""
import runpy
from .supplement_protocol import freeze
from .supplement_models import baseline, feature_experiments
from .supplement_tail import main as tail
from .supplement_statistics import main as statistics
from .review.plot_supplement import main as plots


def main():
    freeze()
    baseline()
    feature_experiments()
    tail()
    statistics()
    runpy.run_module("innovative_solution.review.benchmark_inference",run_name="__main__")
    plots()
    from .supplement_report import generate
    generate()
    from .review.build_response import main as response
    response()


if __name__=="__main__":main()
