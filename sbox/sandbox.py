import re
import tempfile
import subprocess
import json
import os
from jinja2 import Template

module_dir = os.path.dirname(os.path.abspath(__file__))

def extract_function_info(language, input_string):
    if language == 'python':
        function_regex = r"def\s+(.*)\s*\((.*)\)(.*):"
    elif language == 'javascript':
        function_regex = r"function\s+(.*)\s*\((.*)\)(.*){"
    else:
        raise Exception("extract_function_info: Unsupported language")
    
    matches = re.findall(function_regex, input_string, re.MULTILINE)

    functions = []
    for match in matches:
        function_name = match[0]
        arguments = match[1].split(',')

        # Extract argument names by removing any type annotations
        argument_names = [arg.strip().split(':')[0] for arg in arguments if arg]

        function_info = {
            'name': function_name,
            'args': argument_names
        }
        functions.append(function_info)

    return functions

def run_shell_command(command):
    try:
        # Run the shell command and capture its output
        result = subprocess.run(command, shell=True, capture_output=True, text=True)

        # Get the captured output
        output = result.stdout.strip()

        if output == '':
            output = result.stderr.strip()

        # Get the return value
        return_value = result.returncode

        # Return the output and return value
        return output, return_value

    except subprocess.CalledProcessError as e:
        # Handle any errors that occurred during command execution
        print("Error:", e)
        return None, e.returncode
    
class FunctionArg:
    def __init__(self, name, type = None) -> None:
        self.name = name
        self.type = type

class FunctionSandbox:
    def __init__(self, code, language) -> None:
        self.code = code
        self.language = language

        try:
           self.functions = extract_function_info(self.language, self.code)[0]
        except:
           self.functions = { 'name': '', 'args': [] }
        self.name = self.functions['name']
        self.args = [FunctionArg(arg) for arg in self.functions['args']]

        build_out, build_code = run_shell_command(f"cd {module_dir} && docker build . -f Dockerfile.{language} -t sandbox-{language} -q")
        if build_code != 0:
            raise Exception("Error building sandbox docker image:" + build_out)
        
    def build_args(self, args):
        return_args = ''
        for i, arg in enumerate(args):
            if i != 0:
                return_args += ','
            if isinstance(arg, int):
                return_args += str(arg)
            elif isinstance(arg, str):
                return_args += '"'+arg+'"'
            else:
                return_args += str(arg)
        return return_args

    def call(self, *args, **kwargs):
        with open(module_dir+'/eval.'+self.language+'.tpl') as f:
            template = Template(f.read())

        # Create a temporary file
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
            script_file = temp_file.name
            script = template.render(call=self.name+'('+self.build_args(args)+')')
            temp_file.write(script)

        with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
            answer_file = temp_file.name
            temp_file.write(self.code)

        if self.language == "python":
            output, value = run_shell_command(f"docker run -it -v {script_file}:/wrapper.py -v {answer_file}:/answer.py sandbox-python python /wrapper.py")
        elif self.language == "javascript":
            output, value = run_shell_command(f"docker run -it -v {script_file}:/wrapper.js -v {answer_file}:/answer.js sandbox-javascript node /wrapper.js")
       
        start_index = output.find("###")
        if start_index == -1:
            if value != 0:
                return { "error": "non-zero result code "+str(value), "output": output }
            else:
                return output
                
        rv_text = output[start_index + 3:].strip()
        return json.loads(rv_text)
