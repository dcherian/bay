% iterate through and rerun combine_turbulence
% do this after every repo is up-to-date using ganges-pod

clear

rootdir = '/home/deepak/bay/scripts/'
pod.dirs = {'../rama/RAMA13/data/'; ...
            '../rama/RAMA14/data/'; ...
            '../ebob/data/'; };

pod.units = { [526; 527]; ...
              [810; 811; 812; 813; 814]; ...
              [500; 501; 504; 505; 511; 514; 516; 518; 519] };

pod.errors = [];

% select specific units to process here
% e.g. do_units=['526'; '505'] will only process units 526 and 505.
% The rest are skipped
do_units = [];
% do_units = [500, 504, 505, 511];

force = 0;

for dd = 1:length(pod.dirs)
    for uu = 1:size(pod.units{dd}, 1)
        redo_combine = 0;

        if isempty(do_units) | ...
                (~isempty(do_units) & ismember(pod.units{dd}(uu), do_units))

            dirname = [rootdir pod.dirs{dd} num2str(pod.units{dd}(uu)) '/mfiles/'];
            cd(dirname)
            addpath(genpath('./chipod_gust/'))
            hash = githash('driver/combine_turbulence.m');

            % run if Turb.mat doesn't exist, or if Turb.mat hasn't
            % been created with the latest version of the code (as
            % judged by comparing saved commit hash with current
            % commit's hash)
            if ~exist('../proc/Turb.mat', 'file')
                redo_combine = 1;
            else
                load('../proc/Turb.mat')
                if  ~isfield(Turb, 'hash') | ~isequal(hash, Turb.hash)
                    redo_combine = 1;
                end
            end

            if redo_combine | force
                if redo_combine
                    disp(['Unit ' num2str(pod.units{dd}(uu)) ' is out of date.'])
                else
                    disp(['Force-processing unit ' num2str(pod.units{dd}(uu)) '.'])
                end
                % save state because combine_turbulence calls
                % "clear all"
                save('~/do-chipods-state.mat', 'pod', 'dd', 'uu', 'rootdir', ...
                     'do_units', 'force')
                try
                    system('sed -i ''s/on/off/'' combine_turbulence.m')
                    combine_turbulence;
                catch ME
                    system('sed -i ''s/off/on/'' combine_turbulence.m')
                    disp(ME)
                    load('~/do-chipods-state.mat')
                    pod.errors = [pod.errors; pod.units{dd}(uu)]
                end
                system('sed -i ''s/off/on/'' combine_turbulence.m')

                disp('--------------------------------------')
                disp(' ')
                % restore state if necessary
                if ~exist('pod', 'var'), load('~/do-chipods-state.mat'); end
            else
                disp(['Unit ' num2str(pod.units{dd}(uu)) ' is up to date.'])
            end
        end
    end
end

if ~isempty(pod.errors)
    disp('I had trouble with these: ')
    pod.errors
end