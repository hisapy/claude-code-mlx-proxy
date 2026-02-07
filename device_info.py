import mlx.core as mx

if mx.metal.is_available():
    info = mx.metal.device_info()
    
    # This is exactly what you were looking for:
    working_set = info['max_recommended_working_set_size'] / (1024**2)
    total_vram = info['memory_size'] / (1024**2)
    
    print(f"Device: {info['device_name']}")
    print(f"Recommended Max Working Set: {working_set:.2f} MB")
    print(f"Total Unified Memory: {total_vram:.2f} MB")
