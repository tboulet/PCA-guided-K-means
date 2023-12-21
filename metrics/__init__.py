from typing import Dict, Type
from metrics.base_metric import BaseMetric
from metrics.distortion import DistortionMetric
from metrics.number_of_iterations import NumberOfIterationsForConvergenceMetric
from metrics.silhouette import SilhouetteScoreMetric
from metrics.daviesbouldin import DaviesBouldinMetric
from metrics.calinskiharabasz import CalinskiHarabaszMetric
from metrics.confusion import ConfusionMatrixMetric



# This maps the name of the metrics to the class of the metrics.
metrics_name_to_MetricsClass : Dict[str, Type[BaseMetric]] = {
    "distortion" : DistortionMetric,
    "number_of_iterations" : NumberOfIterationsForConvergenceMetric,
    "silhouette" : SilhouetteScoreMetric,
    "davies_bouldin" : DaviesBouldinMetric,
    "calinski_harabasz" : CalinskiHarabaszMetric,
    # "confusion" : ConfusionMatrixMetric,
}
