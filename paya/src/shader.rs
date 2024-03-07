use std::{
    collections::{HashMap, HashSet},
    path::PathBuf,
};

use regex::{Captures, Regex};

use crate::preamble;

pub struct ShaderInfo {
    pub byte_code: Vec<u32>,
    pub entry_point: String,
}

pub struct Shader {}

pub enum ShaderType {
    Compute,
    Vertex,
    Geometry,
    Fragment,
}

pub enum ShaderOptimization {
    None,
    Performance,
    Size,
}

pub struct ShaderCompiler {
    compiler: shaderc::Compiler,
}

pub struct ShaderLoadOptions {
    shader_type: ShaderType,
    optimization: ShaderOptimization,
    entry_point: String,
    name: String,
}

impl ShaderCompiler {
    pub fn new() -> Self {
        Self {
            compiler: shaderc::Compiler::new().expect("Failed to create shaderc compiler."),
        }
    }

    pub fn load_string(&self, shader_source: String, load_options: ShaderLoadOptions) -> Vec<u32> {
        let shader_kind = match load_options.shader_type {
            ShaderType::Compute => shaderc::ShaderKind::Compute,
            ShaderType::Vertex => shaderc::ShaderKind::Vertex,
            ShaderType::Geometry => shaderc::ShaderKind::Geometry,
            ShaderType::Fragment => shaderc::ShaderKind::Fragment,
        };

        let mut options =
            shaderc::CompileOptions::new().expect("Failed to create shaderc compile options.");
        options.set_optimization_level(match load_options.optimization {
            ShaderOptimization::None => shaderc::OptimizationLevel::Zero,
            ShaderOptimization::Performance => shaderc::OptimizationLevel::Performance,
            ShaderOptimization::Size => shaderc::OptimizationLevel::Size,
        });

        let final_source = preamble::SHADER_PREAMBLE_GLSL.to_string() + &shader_source;

        let code_result = self.compiler.compile_into_spirv(
            &final_source,
            shader_kind,
            &load_options.name,
            &load_options.entry_point,
            Some(&options),
        );

        if code_result.is_err() {
            println!("{}", code_result.err().unwrap());
            panic!();
        }

        code_result.unwrap().as_binary().into()
    }

    /// Loads the glsl file and parses includes with relative paths.
    pub fn load_from_file(&self, file_path: String) -> Vec<u32> {
        let root_path = PathBuf::from(file_path.clone());
        let root_path_string = root_path.as_os_str().to_owned().into_string().unwrap();
        let include_regex = Regex::new(r##"#include "([^\"]*\/)*[^"]+""##).unwrap();
        let string_regex = Regex::new(r#""[^"]+""#).unwrap();

        let parse_include_capture = |capture: &Captures, contents: &str, current_file_dir: &str| {
            let m = capture.get(0).unwrap();
            let substr = &contents[m.start()..m.end()];
            let string_match = string_regex.find(substr).unwrap();

            let mut root_dir_string = current_file_dir.to_owned();
            root_dir_string.push_str("/");
            root_dir_string.push_str(&substr[(string_match.start() + 1)..(string_match.end() - 1)]);
            root_dir_string
        };

        let mut cached_files = HashMap::new();
        let mut processed_files: HashMap<String, String> = HashMap::new();
        let mut to_process_files = vec![root_path_string.clone()];
        let mut visited_files = HashSet::new();
        visited_files.insert(root_path_string.clone());

        while !to_process_files.is_empty() {
            let current_file = to_process_files.last().unwrap().clone();
            let current_file_dir = PathBuf::from(current_file.clone())
                .parent()
                .unwrap()
                .as_os_str()
                .to_owned()
                .into_string()
                .unwrap();
            let contents = cached_files.entry(current_file.clone()).or_insert_with(|| {
                std::fs::read_to_string(PathBuf::from(current_file.clone()))
                    .expect(&format!("Could not read file {}", &current_file))
            });

            let mut needs_deps = false;
            for capture in include_regex.captures_iter(contents) {
                let full_path = parse_include_capture(&capture, contents, &current_file_dir);

                if !processed_files.contains_key(&full_path) {
                    if visited_files.contains(&full_path) {
                        panic!(
                            "acyclic deps are not allowed, tried including {}",
                            full_path
                        );
                    }
                    visited_files.insert(full_path.clone());
                    to_process_files.push(full_path);
                    needs_deps = true;
                }
            }

            if !needs_deps {
                let processed_contents = include_regex
                    .replace_all(&contents, |capture: &Captures| {
                        let full_path =
                            parse_include_capture(capture, &contents, &current_file_dir);
                        processed_files.get(&full_path).unwrap().clone()
                    })
                    .to_string();

                to_process_files.pop();
                processed_files.insert(current_file, processed_contents);
            }
        }

        let file_contents = processed_files.get(&root_path_string).unwrap().to_owned();
        let mut file_extension = file_path.split('.').last().unwrap();
        if file_extension == "glsl" {
            file_extension = file_path.split('.').nth_back(1).unwrap();
        }

        let shader_type = match file_extension {
            "vert" => ShaderType::Vertex,
            "geom" => ShaderType::Geometry,
            "frag" => ShaderType::Fragment,
            "comp" => ShaderType::Compute,
            _ => panic!("Glsl sub-extension not supported"),
        };

        self.load_string(
            file_contents,
            ShaderLoadOptions {
                shader_type,
                entry_point: "main".to_owned(),
                name: root_path
                    .file_name()
                    .unwrap()
                    .to_owned()
                    .into_string()
                    .unwrap(),
                optimization: ShaderOptimization::None,
            },
        )
    }
}
