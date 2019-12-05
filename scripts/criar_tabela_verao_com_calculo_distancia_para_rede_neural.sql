create table verao_solo_32011_2016_valfride as

select i6.*,
 (select i1.id from verao_solo_2012 i1 order by  sqrt((i1.long-i6.long)*(i1.long-i6.long)+(i1.lat-i6.lat)*(i1.lat-i6.lat))+abs((extract(epoch from (i6.hora-i1.hora)))), i1.id limit 1) as d1,
 (select i1.id from verao_solo_2012 i1 order by sqrt((i1.long-i6.long)*(i1.long-i6.long)+(i1.lat-i6.lat)*(i1.lat-i6.lat))+abs((extract(epoch from (i6.hora-i1.hora)))), i1.id limit 1 offset 1) as d2,
 (select i1.id from verao_solo_2012 i1  order by  sqrt((i1.long-i6.long)*(i1.long-i6.long)+(i1.lat-i6.lat)*(i1.lat-i6.lat))+abs((extract(epoch from (i6.hora-i1.hora)))), i1.id limit 1 offset 2)  as d3 from verao_solo_2016 i6 order by(data, hora);

