

################## Custom Config ##############
# enable colored prompt for git branches
# using parse_git_branch
parse_git_branch() {
 git branch 2> /dev/null | sed -e '/^[^*]/d' -e 's/* \(.*\)/(\1)/'
}

# modify the PS1 variable to include the parse_git_branch
# \w = full path of current dir, \W = only the name of current dir
# bash color ansi code: https://gist.github.com/iamnewton/8754917
PS1='[\u@\h:\w]$\[\e[96m\]$(parse_git_branch)\[\e[00m\] '

