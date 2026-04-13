import pytest
import json
from backend.proto_out.tricked_pb2 import MetricRow, MetricHistory

def test_metric_row_dimension_mutations():
    row = MetricRow()
    row.step = 100
    row.total_loss = 2.5
    row.spatial_heatmap.extend([1.0, 2.0, 3.0])
    
    assert row.step == 100
    assert row.total_loss == 2.5
    assert len(row.spatial_heatmap) == 3
    
    # Try invalid assignment type (assigning str to float)
    with pytest.raises(TypeError):
        row.total_loss = "invalid_string"
        
    payload = MetricHistory()
    payload.metrics.append(row)
    
    binary = payload.SerializeToString()
    assert isinstance(binary, bytes)
    assert len(binary) > 0
    
    # Decode back
    decoded = MetricHistory()
    decoded.ParseFromString(binary)
    assert decoded.metrics[0].step == 100
    assert len(decoded.metrics[0].spatial_heatmap) == 3
