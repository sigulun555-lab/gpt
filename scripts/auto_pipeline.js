'use strict';
const fs=require('node:fs'),path=require('node:path');
const {predict,backtest}=require('../model.js');
const ROOT=path.resolve(__dirname,'..'),DATA=path.join(ROOT,'data');
const SCHEDULE_URL=process.env.JLEAGUE_CSV_URL||'https://raw.githubusercontent.com/k-addtimes/jleague_open_data/main/current_schedule_file.csv';
const MAX_FIXTURES=13;
const TOTO_URL=process.env.TOTO_PAGE_URL||'https://sp.toto-dream.com/dcs/subos/screen/ss01/sssl014/PGSSSL01401InittotoSP.form';
function csv(text){const out=[];let row=[],cell='',q=false;text=text.replace(/^\uFEFF/,'');for(let i=0;i<text.length;i++){const c=text[i];if(c==='"'&&q&&text[i+1]==='"'){cell+='"';i++;}else if(c==='"')q=!q;else if(c===','&&!q){row.push(cell);cell='';}else if((c==='\n'||c==='\r')&&!q){if(c==='\r'&&text[i+1]==='\n')i++;row.push(cell);if(row.some(x=>x.trim()))out.push(row);row=[];cell='';}else cell+=c;}if(cell||row.length){row.push(cell);if(row.some(x=>x.trim()))out.push(row);}return out;}
function date(s){const m=String(s||'').match(/^(\d{4})[/-](\d{1,2})[/-](\d{1,2})/);return m?m[1]+'-'+m[2].padStart(2,'0')+'-'+m[3].padStart(2,'0'):'';}
function score(s){const m=String(s||'').match(/^(\d+)\s*[-–－]\s*(\d+)$/);return m?{home_goals:+m[1],away_goals:+m[2]}:null;}
function parseSchedule(text){return csv(text).map(r=>({date:date(r[3]),competition:r[1]||'',round:r[2]||'',home:(r[5]||'').trim(),score:r[6]||'',away:(r[7]||'').trim(),...score(r[6])})).filter(m=>m.date&&m.home&&m.away&&m.home!==m.away);}
function decode(s){return String(s||'').replace(/&nbsp;|&#160;/gi,' ').replace(/&amp;/gi,'&').replace(/&lt;/gi,'<').replace(/&gt;/gi,'>').replace(/&quot;/gi,'"').replace(/&#39;|&apos;/gi,"'").replace(/&#(\d+);/g,(_,n)=>String.fromCodePoint(+n));}
function textOf(s){return decode(String(s||'').replace(/<script\b[\s\S]*?<\/script>/gi,' ').replace(/<style\b[\s\S]*?<\/style>/gi,' ').replace(/<\/(?:tr|td|th|div|p|li|option|h[1-6])\s*>/gi,' ').replace(/<[^>]*>/g,' ')).replace(/[\u3000\s]+/g,' ').trim();}
function targetDate(md,now){const m=String(md).match(/^(\d{1,2})\/(\d{1,2})$/);if(!m)return '';const tokyo=new Intl.DateTimeFormat('en-CA',{timeZone:'Asia/Tokyo',year:'numeric',month:'2-digit'}).formatToParts(now),p=Object.fromEntries(tokyo.map(x=>[x.type,x.value]));let y=+p.year;if(+m[1]<+p.month)y++;return y+'-'+m[1].padStart(2,'0')+'-'+m[2].padStart(2,'0');}
function parseGoal3TeamFromRow(rowHtml){
 const rowText=textOf(rowHtml);
 if(!/(?:その他データ|\d+(?:\.\d+)?%|\d+(?:\.\d+)?点)/.test(rowText))return null;
 const cells=[...String(rowHtml).matchAll(/<t[dh]\b[^>]*>([\s\S]*?)<\/t[dh]>/gi)].map(m=>textOf(m[1]));
 const ignored=/^(?:日程|予想チーム|マルチ|その他データ|リアルタイム投票状況|データ反映|[0-3]|[0-3]点|0：0点|\d+(?:\.\d+)?%|\d+(?:\.\d+)?点|\d+\/\d+)$/;
 for(const cell of cells){
  if(!cell||/^\d{1,2}\/\d{1,2}$/.test(cell)||ignored.test(cell))continue;
  let candidate=cell.split(/\s*その他データ|\s*\d+(?:\.\d+)?%|\s*\d+(?:\.\d+)?点/)[0].trim();
  candidate=candidate.replace(/^(?:日程|予想チーム|マルチ)\s*/,'').trim();
  if(candidate&&!ignored.test(candidate)&&!/^[-+\d.%：:点口\s]+$/.test(candidate)&&candidate.length<=40)return candidate;
 }
 return null;
}
function parseToto(html,now=new Date()){
 const plain=textOf(html),rm=plain.match(/〖開催回〗\s*第\s*(\d+)\s*回/)||plain.match(/開催回.{0,180}?第\s*(\d+)\s*回/),round=rm?+rm[1]:null;
 const marketType=/toto\s*GOAL\s*3/i.test(plain)?'totoGOAL3':'toto';
 const sale=plain.match(/販売期間\s*(\d{4}[/-]\d{1,2}[/-]\d{1,2}\s+\d{1,2}:\d{2})\s*[～〜~\-]\s*(\d{4}[/-]\d{1,2}[/-]\d{1,2}\s+\d{1,2}:\d{2})/);
 const iso=v=>{const m=String(v||'').match(/(\d{4})[/-](\d{1,2})[/-](\d{1,2})\s+(\d{1,2}):(\d{2})/);return m?m[1]+'-'+m[2].padStart(2,'0')+'-'+m[3].padStart(2,'0')+'T'+m[4].padStart(2,'0')+':'+m[5]+':00+09:00':null;};
 let fixtures=[];
 if(marketType==='totoGOAL3'){
  const teams=[];
  for(const tr of String(html).matchAll(/<tr\b[^>]*>([\s\S]*?)<\/tr>/gi)){
   const rowHtml=tr[1],team=parseGoal3TeamFromRow(rowHtml);
   if(!team)continue;
   const rowText=textOf(rowHtml),dm=rowText.match(/(?:^|\s)(\d{1,2}\/\d{1,2})(?:\s|$)/);
   teams.push({team,dateLabel:dm?dm[1]:''});
  }
  if(teams.length===6){
   for(let i=0;i<3;i++){
    const home=teams[i*2],away=teams[i*2+1],dateLabel=home.dateLabel||away.dateLabel;
    if(!dateLabel||!home.team||!away.team||home.team===away.team){fixtures=[];break;}
    fixtures.push({number:i+1,marketType,date:targetDate(dateLabel,now),dateLabel,home:home.team,away:away.team,source:'toto_official_page',sourceUrl:TOTO_URL});
   }
  }
 }else{
  for(const tr of String(html).matchAll(/<tr\b[^>]*>([\s\S]*?)<\/tr>/gi)){
   const row=textOf(tr[1]),m=row.match(/(?:^|\s)(\d{1,2})\s+(\d{1,2}\/\d{1,2})\b[\s\S]*?([^\s|]+)\s*VS\s*([^\s|]+)/i);
   if(!m)continue;const n=+m[1];if(!Number.isInteger(n)||n<1||n>MAX_FIXTURES)continue;
   const home=m[3].trim(),away=m[4].trim();
   if(home&&away&&home!==away)fixtures.push({number:n,marketType,date:targetDate(m[2],now),dateLabel:m[2],home,away,source:'toto_official_page',sourceUrl:TOTO_URL});
  }
 }
 const unique=[...new Map(fixtures.map(f=>[f.number,f])).values()].sort((a,b)=>a.number-b.number);
 return {round,marketType,deadline:iso(sale&&sale[2]),saleStart:iso(sale&&sale[1]),fixtures:unique,parsedAt:now.toISOString()};
}
async function get(url,accept){const c=new AbortController(),t=setTimeout(()=>c.abort(),20000);try{const r=await fetch(url,{headers:{Accept:accept,'User-Agent':'JLeagueTotoPredictor/0.1 (+scheduled data pipeline)'},signal:c.signal});if(!r.ok)throw Error('HTTP '+r.status+' from '+new URL(url).hostname);const s=await r.text();if(s.length<100)throw Error('Unexpectedly short response from '+url);return s;}finally{clearTimeout(t);}}
function build(scheduleText,totoHtml,now=new Date()){const rows=parseSchedule(scheduleText),history=rows.filter(m=>Number.isInteger(m.home_goals)&&Number.isInteger(m.away_goals)).sort((a,b)=>a.date.localeCompare(b.date)).map(m=>({...m,source:'community_jleague_csv',sourceUrl:SCHEDULE_URL,fetchedAt:now.toISOString()}));const toto=parseToto(totoHtml,now),expectedFixtures=toto.marketType==='totoGOAL3'?3:MAX_FIXTURES,numberingValid=toto.fixtures.map(x=>x.number).sort((a,b)=>a-b).every((v,i)=>v===i+1),countValid=toto.marketType==='totoGOAL3'?toto.fixtures.length===3:(toto.fixtures.length>0&&toto.fixtures.length<=MAX_FIXTURES),confirmed=countValid&&toto.fixtures.length<=expectedFixtures&&new Set(toto.fixtures.map(x=>x.number)).size===toto.fixtures.length&&numberingValid&&toto.fixtures.every(x=>x.date&&x.home&&x.away)&&!!toto.round&&!!toto.deadline&&Date.parse(toto.deadline)>now.getTime();const fixturesPayload={schemaVersion:1,fixtures:confirmed?toto.fixtures.map(f=>({marketType:toto.marketType,matchNumber:f.number,home:f.home,away:f.away,round:toto.round,deadline:toto.deadline})):[]};const warnings=['負傷・欠場・出場停止情報は未取得です。','予測モデルはベースラインであり、的中を保証しません。'];const predictions=confirmed?toto.fixtures.map(f=>({...f,totoRound:toto.round,totoTargetConfirmed:true,prediction:predict(history,f),availabilityNote:'未取得'})):[];if(!confirmed)warnings.unshift('公式totoページから開催回・締切・対象試合を完全確認できないため予想を公開していません。marketType='+toto.marketType+'、解析数='+toto.fixtures.length);const bt=backtest(history);const payload={schemaVersion:1,generatedAt:now.toISOString(),status:confirmed?'predictions_generated':'toto_targets_unconfirmed',liveData:true,predictionsGenerated:confirmed,totoTargetsConfirmed:confirmed,round:confirmed?toto.round:null,deadline:confirmed?toto.deadline:null,saleStart:confirmed?toto.saleStart:null,source:{name:'k-addtimes/jleague_open_data',url:SCHEDULE_URL,kind:'community mirror; not official toto feed'},totoSource:{name:'スポーツくじ公式公開ページ',url:TOTO_URL,lastFetchedAt:toto.parsedAt,marketType:toto.marketType,round:toto.round,deadline:toto.deadline,fixtureCount:toto.fixtures.length,teamTargetCount:toto.marketType==='totoGOAL3'?toto.fixtures.length*2:null},counts:{scheduleRows:rows.length,historyMatches:history.length,totoFixtures:confirmed?toto.fixtures.length:0,predictions:predictions.length},warnings,predictions};const status={status:payload.status,liveData:true,predictionsGenerated:confirmed,totoTargetsConfirmed:confirmed,round:payload.round,deadline:payload.deadline,lastAttemptAt:now.toISOString(),lastSuccessfulFetch:now.toISOString(),lastError:null,source:payload.source,totoSource:payload.totoSource,counts:payload.counts,warnings};return {payload,status,history,backtest:bt,fixtures:fixturesPayload};}
async function main(){fs.mkdirSync(DATA,{recursive:true});const now=new Date();try{const [s,t]=await Promise.all([get(SCHEDULE_URL,'text/csv,text/plain,*/*'),get(TOTO_URL,'text/html,application/xhtml+xml,*/*')]);const r=build(s,t,now);fs.writeFileSync(path.join(DATA,'predictions.json'),JSON.stringify(r.payload,null,2)+'\n');fs.writeFileSync(path.join(DATA,'status.json'),JSON.stringify(r.status,null,2)+'\n');fs.writeFileSync(path.join(DATA,'history.json'),JSON.stringify({history:r.history,source:r.payload.source,updatedAt:now.toISOString(),liveData:true},null,2)+'\n');fs.writeFileSync(path.join(DATA,'backtest.json'),JSON.stringify({...r.backtest,updatedAt:now.toISOString(),source:r.payload.source},null,2)+'\n');fs.writeFileSync(path.join(DATA,'fixtures.json'),JSON.stringify(r.fixtures,null,2)+'\n');console.log(JSON.stringify({status:r.payload.status,counts:r.payload.counts,round:r.payload.round},null,2));if(!r.payload.totoTargetsConfirmed)process.exitCode=2;}catch(e){let old={};try{old=JSON.parse(fs.readFileSync(path.join(DATA,'status.json'),'utf8'));}catch{}const failed={...old,status:'fetch_failed',lastAttemptAt:now.toISOString(),lastError:String(e.message||e),warnings:[...(old.warnings||[]),'最新の取得に失敗しました。予想がある場合は最終成功取得日時を確認してください。']};fs.writeFileSync(path.join(DATA,'status.json'),JSON.stringify(failed,null,2)+'\n');console.error('PIPELINE_FAILED:',failed.lastError);process.exitCode=1;}}
if(require.main===module)main();
module.exports={csv,date,score,parseSchedule,parseGoal3TeamFromRow,parseToto,build};
