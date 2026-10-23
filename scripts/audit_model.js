'use strict';
const fs=require('node:fs'),path=require('node:path');
const {backtest,canonical,leagueOf}=require('../model.js');
const root=path.resolve(__dirname,'..');
const history=JSON.parse(fs.readFileSync(path.join(root,'data/history.json'),'utf8')).history;
const aliases=JSON.parse(fs.readFileSync(path.join(root,'data/team_aliases.json'),'utf8')).aliases;
const keys=history.map(m=>[m.date,m.competition,canonical(m.home),canonical(m.away)].join('|'));
const dup=keys.filter((k,i)=>keys.indexOf(k)!==i);
const missing=history.filter(m=>!m.date||!m.competition||!m.home||!m.away||!Number.isInteger(m.home_goals)||!Number.isInteger(m.away_goals)||m.home_goals<0||m.away_goals<0);
const counts={},unmapped=new Set(),dateRows=history.map(m=>m.date).sort();
for(const m of history){counts[m.competition]=(counts[m.competition]||0)+1;if(!leagueOf(m.home))unmapped.add(m.home);if(!leagueOf(m.away))unmapped.add(m.away);}
const sorted=[...history].sort((a,b)=>a.date.localeCompare(b.date));
const sameDayLeakCandidates=sorted.filter(m=>sorted.some(x=>x.date===m.date&&x!==m));
const report={generatedAt:new Date().toISOString(),historyRows:history.length,dateRange:[dateRows[0],dateRows.at(-1)],competitionCounts:counts,duplicateFixtures:dup.length,invalidRows:missing.length,unmappedTeams:[...unmapped].sort(),sameDayGroups:Object.values(history.reduce((a,m)=>(a[m.date]=(a[m.date]||0)+1,a),{})).filter(n=>n>1).length,backtest:{leagueFiltered:backtest(history),mixedCompetitions:backtest(history,{filterCompetition:false})},notes:['Same-day fixtures are excluded from each prediction by strict date-before-fixture filtering.','Historical source is a community mirror, not an official results feed.','Backtest uses walk-forward predictions; results should be interpreted by league and season, not as a guarantee of future accuracy.']};
console.log(JSON.stringify(report,null,2));
if(dup.length||missing.length||unmapped.size)process.exitCode=1;
