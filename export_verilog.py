#!/usr/bin/env python3
"""
Export OpenTPU PyRTL design to Verilog.

This script generates structural Verilog from the PyRTL hardware description.
The Verilog can be used for synthesis, simulation, or further analysis.

Usage:
    python3 export_verilog.py [--output OUTPUT_FILE] [--optimize] [--synthesize] [--module-name NAME]
"""

import argparse
import sys
import os

# Check for PyRTL
try:
    from pyrtl import output_to_verilog, synthesize, optimize
except ImportError:
    print("Error: PyRTL is not installed.")
    print("Please install it with: pip install pyrtl")
    print("\nNote: If you're using a virtual environment, make sure it's activated.")
    print("      You can activate it with: source .tpu/bin/activate")
    sys.exit(1)

# Import the TPU design - this builds the hardware description
# Note: We need to import this AFTER argparse setup to avoid issues
# but the actual import happens when we execute the import statement below

def export_verilog(output_file='tpu.v', optimize_design=False, 
                   synthesize_design=False, module_name='tpu', 
                   add_reset=True, initialize_registers=False):
    """
    Export the TPU design to Verilog.
    
    Args:
        output_file: Path to output Verilog file
        optimize_design: Whether to run optimization before export
        synthesize_design: Whether to run synthesis before export
        module_name: Name of the Verilog module
        add_reset: Whether to add reset logic (True, False, or 'asynchronous')
        initialize_registers: Whether to initialize registers to reset values
    """
    print("=" * 70)
    print("OpenTPU Verilog Export")
    print("=" * 70)
    
    # Import the TPU design - this executes tpu.py and builds the hardware
    print("\n[1/4] Importing TPU design...")
    try:
        # #region agent log - Initialize logging early
        import json
        import time
        import os
        log_dir = os.path.dirname('/home/herbelinluke/Documents/ersp/OpenTPU/.cursor/debug.log')
        os.makedirs(log_dir, exist_ok=True)
        with open('/home/herbelinluke/Documents/ersp/OpenTPU/.cursor/debug.log', 'a') as logf:
            logf.write(json.dumps({
                'sessionId': 'debug-session', 'runId': 'post-fix', 'hypothesisId': 'A',
                'location': 'export_verilog.py:47', 'message': 'Starting TPU import',
                'data': {},
                'timestamp': int(time.time() * 1000)
            }) + '\n')
        # #endregion
        import tpu
        # #region agent log - Check I/O after tpu import
        from pyrtl import working_block, Input, Output
        block = working_block()
        all_wires = list(block.wirevector_set)
        input_wires = [w for w in all_wires if isinstance(w, Input)]
        output_wires = [w for w in all_wires if isinstance(w, Output)]
        with open('/home/herbelinluke/Documents/ersp/OpenTPU/.cursor/debug.log', 'a') as logf:
            logf.write(json.dumps({
                'sessionId': 'debug-session', 'runId': 'post-fix', 'hypothesisId': 'A',
                'location': 'export_verilog.py:56', 'message': 'I/O check after tpu import',
                'data': {
                    'input_count': len(input_wires), 'output_count': len(output_wires),
                    'input_details': [{'name': getattr(w, 'name', None), 'has_name': hasattr(w, 'name'), 'name_type': type(getattr(w, 'name', None)).__name__ if hasattr(w, 'name') and getattr(w, 'name', None) is not None else 'N/A'} for w in input_wires],
                    'output_details': [{'name': getattr(w, 'name', None), 'has_name': hasattr(w, 'name'), 'name_type': type(getattr(w, 'name', None)).__name__ if hasattr(w, 'name') and getattr(w, 'name', None) is not None else 'N/A'} for w in output_wires]
                },
                'timestamp': int(time.time() * 1000)
            }) + '\n')
        # #endregion
        print("    ✓ TPU design imported successfully")
    except NameError as e:
        if 'helperfuncs' in str(e):
            print(f"    ✗ Error: {e}")
            print("    This is a known issue - helperfuncs.mult_signed should be replaced with signed_mult")
            print("    Please check that matrix.py has been updated.")
            sys.exit(1)
        else:
            print(f"    ✗ Error importing TPU design: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    except Exception as e:
        print(f"    ✗ Error importing TPU design: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Synthesize if requested
    if synthesize_design:
        print("\n[2/4] Synthesizing design...")
        try:
            synthesize()
            print("    ✓ Synthesis complete")
        except Exception as e:
            print(f"    ✗ Error during synthesis: {e}")
            sys.exit(1)
    else:
        print("\n[2/4] Skipping synthesis (use --synthesize to enable)")
    
    # Optimize if requested
    if optimize_design:
        print("\n[3/4] Optimizing design...")
        try:
            optimize()
            print("    ✓ Optimization complete")
        except Exception as e:
            print(f"    ✗ Error during optimization: {e}")
            sys.exit(1)
    else:
        print("\n[3/4] Skipping optimization (use --optimize to enable)")
    
    # Export to Verilog
    print(f"\n[4/4] Exporting to Verilog: {output_file}")
    # #region agent log - Final check before export attempt
    import json
    import time
    import os
    from pyrtl import working_block, Input, Output
    block = working_block()
    all_wires = list(block.wirevector_set)
    input_wires = [w for w in all_wires if isinstance(w, Input)]
    output_wires = [w for w in all_wires if isinstance(w, Output)]
    log_dir = os.path.dirname('/home/herbelinluke/Documents/ersp/OpenTPU/.cursor/debug.log')
    os.makedirs(log_dir, exist_ok=True)
    try:
        with open('/home/herbelinluke/Documents/ersp/OpenTPU/.cursor/debug.log', 'a') as logf:
            logf.write(json.dumps({
                'sessionId': 'debug-session', 'runId': 'post-fix', 'hypothesisId': 'C',
                'location': 'export_verilog.py:117', 'message': 'Final I/O check before export',
                'data': {
                    'input_count': len(input_wires), 'output_count': len(output_wires),
                    'all_input_names': [str(getattr(w, 'name', 'NO_NAME')) for w in input_wires],
                    'all_output_names': [str(getattr(w, 'name', 'NO_NAME')) for w in output_wires],
                    'inputs_with_tmp': [str(getattr(w, 'name', '')) for w in input_wires if getattr(w, 'name', '').startswith('tmp')],
                    'outputs_with_tmp': [str(getattr(w, 'name', '')) for w in output_wires if getattr(w, 'name', '').startswith('tmp')]
                },
                'timestamp': int(time.time() * 1000)
            }) + '\n')
    except Exception as log_err:
        print(f"    ⚠ Warning: Could not write to debug log: {log_err}", file=sys.stderr)
        # Fallback: write to stderr
        import json
        print(json.dumps({
            'error': 'log_write_failed',
            'message': str(log_err),
            'input_count': len(input_wires),
            'output_count': len(output_wires)
        }), file=sys.stderr)
    # #endregion
    try:
        # #region agent log - Verify all I/O have names before export and fix any missing names
        import json
        import time
        import os
        from pyrtl import working_block, Input, Output
        block = working_block()
        all_wires = list(block.wirevector_set)
        input_wires = [w for w in all_wires if isinstance(w, Input)]
        output_wires = [w for w in all_wires if isinstance(w, Output)]
        
        # Fix any Input/Output wires without names or with tmp names
        # Also ensure names are valid Verilog identifiers (start with letter/underscore, alphanumeric+underscore only)
        import re
        verilog_name_regex = re.compile(r'^[_A-Za-z][_a-zA-Z0-9]*$')
        input_counter = 0
        output_counter = 0
        for w in input_wires:
            try:
                wire_name = getattr(w, 'name', None)
                # Fix if: no name, empty name, starts with tmp, or invalid Verilog identifier
                needs_fix = (not wire_name or wire_name == "" or 
                            wire_name.startswith('tmp') or 
                            not verilog_name_regex.match(wire_name))
                if needs_fix:
                    new_name = f"tpu_input_{input_counter}"
                    w.name = new_name
                    input_counter += 1
            except Exception as e:
                # If we can't set the name, log it but continue
                pass
        for w in output_wires:
            try:
                wire_name = getattr(w, 'name', None)
                # Fix if: no name, empty name, starts with tmp, or invalid Verilog identifier
                needs_fix = (not wire_name or wire_name == "" or 
                            wire_name.startswith('tmp') or 
                            not verilog_name_regex.match(wire_name))
                if needs_fix:
                    new_name = f"tpu_output_{output_counter}"
                    w.name = new_name
                    output_counter += 1
            except Exception as e:
                # If we can't set the name, log it but continue
                pass
        
        # Log the state after fixing
        tmp_inputs = [w for w in input_wires if hasattr(w, 'name') and w.name and w.name.startswith('tmp')]
        tmp_outputs = [w for w in output_wires if hasattr(w, 'name') and w.name and w.name.startswith('tmp')]
        empty_name_inputs = [w for w in input_wires if not hasattr(w, 'name') or not w.name]
        empty_name_outputs = [w for w in output_wires if not hasattr(w, 'name') or not w.name]
        log_data = {
            'sessionId': 'debug-session', 'runId': 'post-fix', 'hypothesisId': 'A',
            'location': 'export_verilog.py:94', 'message': 'I/O signal verification and fix before export',
            'data': {
                'input_count': len(input_wires), 'output_count': len(output_wires),
                'tmp_inputs_after_fix': len(tmp_inputs), 'tmp_outputs_after_fix': len(tmp_outputs),
                'empty_name_inputs_after_fix': len(empty_name_inputs), 'empty_name_outputs_after_fix': len(empty_name_outputs),
                'all_input_names': [getattr(w, 'name', 'NO_NAME') for w in input_wires],
                'all_output_names': [getattr(w, 'name', 'NO_NAME') for w in output_wires],
                'fixed_inputs': input_counter, 'fixed_outputs': output_counter
            },
            'timestamp': int(time.time() * 1000)
        }
        # Ensure log directory exists
        log_dir = os.path.dirname('/home/herbelinluke/Documents/ersp/OpenTPU/.cursor/debug.log')
        os.makedirs(log_dir, exist_ok=True)
        with open('/home/herbelinluke/Documents/ersp/OpenTPU/.cursor/debug.log', 'a') as logf:
            logf.write(json.dumps(log_data) + '\n')
        # #endregion
        
        # #region agent log - Check what PyRTL will see during export
        # Simulate what PyRTL does - check the gate graph
        from pyrtl.importexport import GateGraph, _VerilogSanitizer
        gate_graph = GateGraph(block)
        sanitizer = _VerilogSanitizer("_ver_out_tmp_")
        io_list_simulated = ["clk"]
        for inp in gate_graph.inputs:
            if inp.name:
                sanitized = sanitizer.make_valid_string(inp.name)
                io_list_simulated.append(sanitized)
        for out in gate_graph.outputs:
            if out.name:
                sanitized = sanitizer.make_valid_string(out.name)
                io_list_simulated.append(sanitized)
        tmp_names = [name for name in io_list_simulated if name.startswith('tmp')]
        with open('/home/herbelinluke/Documents/ersp/OpenTPU/.cursor/debug.log', 'a') as logf:
            logf.write(json.dumps({
                'sessionId': 'debug-session', 'runId': 'post-fix', 'hypothesisId': 'B',
                'location': 'export_verilog.py:155', 'message': 'Simulated PyRTL io_list check',
                'data': {
                    'io_list_simulated': io_list_simulated,
                    'tmp_names_found': tmp_names,
                    'gate_graph_inputs': [{'name': getattr(g, 'name', None), 'type': str(type(g))} for g in gate_graph.inputs],
                    'gate_graph_outputs': [{'name': getattr(g, 'name', None), 'type': str(type(g))} for g in gate_graph.outputs]
                },
                'timestamp': int(time.time() * 1000)
            }) + '\n')
        # #endregion
        
        # #region agent log - Final verification before export call
        # Double-check all I/O have valid names one more time right before export
        final_inputs = [w for w in input_wires]
        final_outputs = [w for w in output_wires]
        # Ensure all I/O wires have valid names before export
        for w in final_inputs + final_outputs:
            try:
                name = getattr(w, 'name', None)
                if not name or name == "" or name.startswith('tmp'):
                    # Generate a safe name
                    if isinstance(w, Input):
                        new_name = f"fixed_input_{id(w) % 10000}"
                    else:
                        new_name = f"fixed_output_{id(w) % 10000}"
                    w.name = new_name
            except Exception:
                pass  # If we can't set the name, continue anyway
        # #endregion
        
        with open(output_file, 'w') as f:
            # #region agent log - Catch error during export
            try:
                # Note: PyRTL's output_to_verilog doesn't support module_name parameter
                # in all versions. We'll export with default name and post-process if needed.
                output_to_verilog(
                    dest_file=f,
                    add_reset=add_reset,
                    initialize_registers=initialize_registers
                )
            except Exception as export_error:
                import traceback
                error_tb = traceback.format_exc()
                error_msg = str(export_error)
                # Try to write to log
                try:
                    with open('/home/herbelinluke/Documents/ersp/OpenTPU/.cursor/debug.log', 'a') as logf:
                        logf.write(json.dumps({
                            'sessionId': 'debug-session', 'runId': 'post-fix', 'hypothesisId': 'A',
                            'location': 'export_verilog.py:245', 'message': 'Export error caught',
                            'data': {
                                'error_type': type(export_error).__name__,
                                'error_message': error_msg,
                                'traceback': error_tb,
                                'final_input_names': [getattr(w, 'name', 'NO_NAME') for w in final_inputs],
                                'final_output_names': [getattr(w, 'name', 'NO_NAME') for w in final_outputs]
                            },
                            'timestamp': int(time.time() * 1000)
                        }) + '\n')
                except:
                    # Fallback: print to stderr
                    print(f"DEBUG: Export error: {error_msg}", file=sys.stderr)
                    print(f"DEBUG: Input names: {[getattr(w, 'name', 'NO_NAME') for w in final_inputs]}", file=sys.stderr)
                    print(f"DEBUG: Output names: {[getattr(w, 'name', 'NO_NAME') for w in final_outputs]}", file=sys.stderr)
                
                # If it's the "tmp" name error, try one more fix and retry
                if 'tmp' in error_msg.lower() and 'unnamed' in error_msg.lower():
                    print("    ⚠ Detected 'tmp' name error, attempting automatic fix...")
                    # Force fix all I/O names one more time
                    for i, w in enumerate(final_inputs):
                        try:
                            w.name = f"tpu_in_{i}"
                        except:
                            pass
                    for i, w in enumerate(final_outputs):
                        try:
                            w.name = f"tpu_out_{i}"
                        except:
                            pass
                    # Retry export
                    try:
                        f.seek(0)
                        f.truncate()
                        output_to_verilog(
                            dest_file=f,
                            add_reset=add_reset,
                            initialize_registers=initialize_registers
                        )
                        print("    ✓ Export succeeded after automatic fix")
                    except Exception as retry_error:
                        print(f"    ✗ Retry also failed: {retry_error}")
                        raise export_error  # Raise original error
                else:
                    raise
            # #endregion
        
        # If a custom module name was requested, do post-processing to rename it
        if module_name != 'toplevel':
            try:
                with open(output_file, 'r') as f:
                    content = f.read()
                # Replace module declaration (handle both with and without parameters)
                content = content.replace('module toplevel(', f'module {module_name}(')
                content = content.replace('module toplevel;', f'module {module_name};')
                # Also replace in comments that reference the module
                content = content.replace('synth_xilinx -top toplevel', f'synth_xilinx -top {module_name}')
                with open(output_file, 'w') as f:
                    f.write(content)
                print(f"    ✓ Module name changed to '{module_name}' via post-processing")
            except Exception as e:
                print(f"    ⚠ Warning: Could not change module name: {e}")
                print(f"    Module will remain as 'toplevel'")
        
        print(f"    ✓ Verilog exported successfully to {output_file}")
    except Exception as e:
        print(f"    ✗ Error exporting Verilog: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("\n" + "=" * 70)
    print("Export complete!")
    print("=" * 70)
    print(f"\nGenerated Verilog module: {module_name}")
    print(f"Output file: {output_file}")
    print("\nNote: The Verilog is structural (gate-level) and may be very large.")
    print("      For synthesis, you may want to use --synthesize and --optimize flags.")


def main():
    parser = argparse.ArgumentParser(
        description='Export OpenTPU PyRTL design to Verilog',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic export
  python3 export_verilog.py
  
  # Export with optimization and synthesis
  python3 export_verilog.py --optimize --synthesize
  
  # Custom output file and module name
  python3 export_verilog.py -o my_tpu.v --module-name my_tpu
  
  # Export with asynchronous reset
  python3 export_verilog.py --reset-async
        """
    )
    
    parser.add_argument(
        '-o', '--output',
        default='tpu.v',
        help='Output Verilog file path (default: tpu.v)'
    )
    
    parser.add_argument(
        '--optimize',
        action='store_true',
        help='Run optimization before export (recommended for synthesis)'
    )
    
    parser.add_argument(
        '--synthesize',
        action='store_true',
        help='Run synthesis before export (converts to gates, recommended for synthesis)'
    )
    
    parser.add_argument(
        '--module-name',
        default='tpu',
        help='Name of the Verilog module (default: tpu)'
    )
    
    parser.add_argument(
        '--no-reset',
        action='store_true',
        help='Do not add reset logic to the Verilog output'
    )
    
    parser.add_argument(
        '--reset-async',
        action='store_true',
        help='Use asynchronous reset instead of synchronous reset'
    )
    
    parser.add_argument(
        '--initialize-registers',
        action='store_true',
        help='Initialize Verilog registers to their reset values'
    )
    
    args = parser.parse_args()
    
    # Determine reset type
    if args.no_reset:
        reset_type = False
    elif args.reset_async:
        reset_type = 'asynchronous'
    else:
        reset_type = True  # default synchronous reset
    
    # Export Verilog
    export_verilog(
        output_file=args.output,
        optimize_design=args.optimize,
        synthesize_design=args.synthesize,
        module_name=args.module_name,
        add_reset=reset_type,
        initialize_registers=args.initialize_registers
    )


if __name__ == '__main__':
    main()
