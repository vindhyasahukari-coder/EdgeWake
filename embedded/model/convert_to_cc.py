import os

def convert_tflite_to_c_array(tflite_path, output_path, array_name="g_kws_model_data"):
    if not os.path.exists(tflite_path):
        print(f"Error: {tflite_path} not found.")
        return

    with open(tflite_path, "rb") as f:
        data = f.read()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, "w") as f:
        f.write(f"// Automatically generated C++ array from {os.path.basename(tflite_path)}\n")
        f.write(f"alignas(8) const unsigned char {array_name}[] = {{\n")
        
        # Write hex bytes
        for i, byte in enumerate(data):
            if i % 12 == 0:
                f.write("    ")
            f.write(f"0x{byte:02x}, ")
            if (i + 1) % 12 == 0:
                f.write("\n")
                
        f.write("\n};\n")
        f.write(f"const int {array_name}_len = {len(data)};\n")
        
    print(f"Successfully converted to {output_path}")

if __name__ == "__main__":
    tflite_model = "models/kws_model_int8.tflite"
    output_cc = "embedded/model/kws_model_data.cc"
    convert_tflite_to_c_array(tflite_model, output_cc)
