"""Customer Segmentation: RFM + K-means, UCI Online Retail (dataset 352)."""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
COLUMNS = {'InvoiceNo','InvoiceDate','CustomerID','Quantity','UnitPrice'}


def load_data() -> pd.DataFrame:
    df = pd.read_excel(ROOT / 'data' / 'Online Retail.xlsx')
    if not COLUMNS.issubset(df.columns): raise ValueError(f'Немає стовпців: {COLUMNS-set(df.columns)}')
    return df


def clean_transactions(df: pd.DataFrame) -> tuple[pd.DataFrame,dict]:
    d=df.copy()
    report={'raw_rows':len(d),'missing_customer_id':int(d.CustomerID.isna().sum())}
    d['InvoiceNo']=d.InvoiceNo.astype('string').str.strip()
    d['InvoiceDate']=pd.to_datetime(d.InvoiceDate,errors='coerce')
    for col in ('Quantity','UnitPrice'): d[col]=pd.to_numeric(d[col],errors='coerce')
    d=d.dropna(subset=['CustomerID','InvoiceNo','InvoiceDate','Quantity','UnitPrice'])
    report['cancellation_rows']=int(d.InvoiceNo.str.upper().str.startswith('C',na=False).sum())
    d=d.loc[~d.InvoiceNo.str.upper().str.startswith('C',na=False)]
    report['nonpositive_quantity_or_price']=int(((d.Quantity<=0)|(d.UnitPrice<=0)).sum())
    d=d.loc[(d.Quantity>0)&(d.UnitPrice>0)].copy()
    d['CustomerID']=d.CustomerID.astype(float).astype(int).astype(str)
    d['TotalPrice']=d.Quantity*d.UnitPrice
    report['clean_rows']=len(d)
    report['unique_customers']=int(d.CustomerID.nunique())
    report['unique_invoices']=int(d.InvoiceNo.nunique())
    return d,report


def calculate_rfm(df: pd.DataFrame) -> tuple[pd.DataFrame,pd.Timestamp]:
    if df.empty: raise ValueError('Немає валідних транзакцій')
    snapshot=df.InvoiceDate.max().normalize()+pd.Timedelta(days=1)
    result=df.groupby('CustomerID').agg(
        LastPurchase=('InvoiceDate','max'),
        Frequency=('InvoiceNo','nunique'),
        Monetary=('TotalPrice','sum'))
    result['Recency']=(snapshot-result.LastPurchase.dt.normalize()).dt.days
    result=result[['Recency','Frequency','Monetary']].sort_index()
    if (result<=0).any().any(): raise ValueError('RFM повинні бути додатними')
    return result,snapshot


def prepare_features(rfm: pd.DataFrame) -> tuple[np.ndarray,StandardScaler]:
    # log1p reduces the effect of very large (wholesale) purchases.
    transformed=np.log1p(rfm[['Recency','Frequency','Monetary']].to_numpy(dtype=float))
    scaler=StandardScaler()
    return scaler.fit_transform(transformed),scaler


def pick_k(x: np.ndarray, ks: range,seed: int=42) -> tuple[int,pd.DataFrame]:
    rows=[]
    for k in ks:
        model=KMeans(n_clusters=k,random_state=seed,n_init=10).fit(x)
        # A fixed random sample for reproducibility, important for large datasets.
        silhouette=silhouette_score(x,model.labels_,sample_size=min(2500,len(x)),random_state=seed)
        rows.append({'k':k,'inertia':model.inertia_,'silhouette':silhouette})
    score=pd.DataFrame(rows)
    best=int(score.loc[score.silhouette.idxmax(),'k'])
    return best,score


def profile_clusters(rfm:pd.DataFrame) -> pd.DataFrame:
    g=rfm.groupby('Cluster',sort=True)
    result=g.agg(Customers=('Recency','size'),Recency_median=('Recency','median'),
                 Frequency_median=('Frequency','median'),Monetary_median=('Monetary','median'),
                 Monetary_total=('Monetary','sum')).round(2)
    result['Share_percent']=(100*result.Customers/len(rfm)).round(2)
    result['Revenue_share_percent']=(100*result.Monetary_total/rfm.Monetary.sum()).round(2)
    return result


def name_clusters(profiles: pd.DataFrame) -> dict[int,str]:
    # Names based on cluster-level RFM, not on arbitrary K-means label numbers.
    # These are interpretive labels, not ground-truth customer classes.
    ranks=pd.DataFrame(index=profiles.index)
    ranks['R']=profiles.Recency_median.rank(ascending=False,pct=True)
    ranks['F']=profiles.Frequency_median.rank(ascending=True,pct=True)
    ranks['M']=profiles.Monetary_median.rank(ascending=True,pct=True)
    score=(ranks.R+ranks.F+ranks.M)/3
    names={}
    ordered=score.sort_values().index.to_list()
    for i,c in enumerate(ordered):
        if i==0: label='At risk / low engagement'
        elif i==len(ordered)-1: label='High-value loyal'
        elif ranks.loc[c,'R']<=.4: label='Inactive / occasional'
        elif ranks.loc[c,'F']>=.6 and ranks.loc[c,'M']>=.6: label='Regular valuable'
        else: label='Developing / occasional'
        names[int(c)]=label
    return names


def charts(rfm,x,score,profiles,output):
    plt.rcParams.update({'figure.dpi':130,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,3,figsize=(13,3.6))
    for ax,col,clr in zip(axes,['Recency','Frequency','Monetary'],['#2B6777','#52AB98','#D38A53']):
        ax.hist(np.log1p(rfm[col]),bins=40,color=clr,edgecolor='white',alpha=.9)
        ax.set(xlabel=f'log(1 + {col})',ylabel='Customers',title=col)
    fig.suptitle('RFM feature distributions (log scale)',fontweight='bold');fig.tight_layout();fig.savefig(output/'01_rfm_distributions.png');plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,3.6))
    for ax,(a,b) in zip(axes,[('Recency','Frequency'),('Recency','Monetary'),('Frequency','Monetary')]):
        ax.scatter(np.log1p(rfm[a]),np.log1p(rfm[b]),s=8,alpha=.3,color='#2B6777')
        ax.set(xlabel=f'log(1 + {a})',ylabel=f'log(1 + {b})',title=f'{a} vs {b}')
    fig.suptitle('Relationships between RFM features',fontweight='bold');fig.tight_layout();fig.savefig(output/'02_rfm_relationships.png');plt.close(fig)
    fig,ax1=plt.subplots(figsize=(7,4)); ax2=ax1.twinx()
    ax1.plot(score.k,score.inertia,'o-',color='#2B6777',label='Inertia')
    ax2.plot(score.k,score.silhouette,'s--',color='#DB8A48',label='Silhouette')
    ax1.set(xlabel='Number of clusters (k)',ylabel='Inertia',title='K-means: Elbow and Silhouette')
    ax2.set_ylabel('Silhouette coefficient');fig.tight_layout();fig.savefig(output/'03_k_selection.png');plt.close(fig)
    pca=PCA(n_components=2,random_state=42)
    projected=pca.fit_transform(x)
    fig,ax=plt.subplots(figsize=(8,5))
    for c in sorted(rfm.Cluster.unique()):
        mask=rfm.Cluster.to_numpy()==c
        ax.scatter(projected[mask,0],projected[mask,1],s=8,alpha=.40,label=f'Cluster {c}',rasterized=True)
    ax.legend(markerscale=2);ax.set(xlabel=f'PC1 ({pca.explained_variance_ratio_[0]:.1%})',
                                  ylabel=f'PC2 ({pca.explained_variance_ratio_[1]:.1%})',title='Clusters projected with PCA')
    fig.tight_layout();fig.savefig(output/'04_pca_clusters.png');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    profiles.Customers.plot.bar(ax=axes[0],color='#2B6777',rot=0,title='Customer count by cluster',ylabel='Customers')
    profiles.Revenue_share_percent.plot.bar(ax=axes[1],color='#DB8A48',rot=0,title='Revenue by cluster',ylabel='Revenue share, %')
    fig.tight_layout();fig.savefig(output/'05_cluster_sizes_revenue.png');plt.close(fig)
    means=rfm.groupby('Cluster')[['Recency','Frequency','Monetary']].median()
    normalized=np.log1p(means).copy()
    normalized=(normalized-normalized.mean())/normalized.std(ddof=0).replace(0,1)
    fig,ax=plt.subplots(figsize=(7,4))
    img=ax.imshow(normalized,aspect='auto',cmap='RdBu_r',vmin=-2,vmax=2)
    ax.set_xticks(range(3),means.columns);ax.set_yticks(range(len(means)),[f'Cluster {c}' for c in means.index])
    ax.set_title('Cluster median profiles (standardized log scale)')
    fig.colorbar(img,ax=ax,label='Relative to clusters');fig.tight_layout();fig.savefig(output/'06_cluster_heatmap.png');plt.close(fig)
    return pca.explained_variance_ratio_.tolist()


def build_report(data_stats,df,rfm,profiles,score,best,snapshot,pca_ratio,out):
    features=['Recency','Frequency','Monetary']
    descriptions={
        'Recency':'Давність останньої покупки у днях',
        'Frequency':'Кількість унікальних замовлень',
        'Monetary':'Загальна сума покупок у GBP',
    }
    summary=rfm[features].agg(['min','median','mean','max']).T.round(2)
    summary.index.name='Ознака'
    summary.columns=['Мінімум','Медіана','Середнє','Максимум']
    correlation=np.log1p(rfm[features]).corr()
    evaluation=score.copy()
    evaluation['k']=evaluation.k.astype(int).astype(str)
    strongest=profiles.sort_values('Monetary_median',ascending=False).iloc[0]
    strongest_id=int(profiles.Monetary_median.idxmax())
    weakest_id=int(profiles.Monetary_median.idxmin())
    metrics={
        'Customers':'Кількість клієнтів',
        'Share_percent':'Частка клієнтів, %',
        'Recency_median':'Медіана Recency, дні',
        'Frequency_median':'Медіана Frequency, замовлення',
        'Monetary_median':'Медіана Monetary, GBP',
        'Monetary_total':'Сума покупок, GBP',
        'Revenue_share_percent':'Частка виручки, %',
    }
    comparison=profiles[list(metrics)].T.rename(index=metrics)
    comparison.columns=[f'Кластер {int(c)}' for c in comparison.columns]
    comparison.index.name='Показник'
    cluster_text=[]
    for cluster,p in profiles.iterrows():
        cluster_text.append(
            f'**Кластер {int(cluster)} — {p.Segment_name}.** '
            f'Містить {int(p.Customers):,} клієнтів ({p.Share_percent:.2f}% бази). '
            f'Медіанна давність покупки — {p.Recency_median:.0f} днів, '
            f'кількість замовлень — {p.Frequency_median:.0f}, '
            f'сума витрат — £{p.Monetary_median:,.2f}. '
            f'Частка врахованої виручки — {p.Revenue_share_percent:.2f}%.'
        )
    report=f"""# Звіт з сегментації клієнтів методом K-means

**Виконала:** Черниченко Надія, група ТТП-42  
**Тема:** Customer Segmentation на основі RFM-показників  
**Мова реалізації:** Python; основні бібліотеки — pandas і scikit-learn

## Мета роботи

Мета — виділити групи клієнтів зі схожою купівельною поведінкою та пояснити відмінності між ними. Для цього історію транзакцій перетворено на RFM-показники та застосовано кластеризацію K-means. Готових класів клієнтів у даних немає, тому задача належить до навчання без учителя.

У результаті аналізу {len(rfm):,} клієнтів обрано **{best} кластери** із Silhouette Score **{score.loc[score.k==best,'silhouette'].iloc[0]:.4f}**. Група з вищою медіаною витрат становить {strongest.Share_percent:.2f}% клієнтів і забезпечує {strongest.Revenue_share_percent:.2f}% врахованої виручки.

## 1 Попередня обробка даних

### Завантаження та структура

Використано [Online Retail з UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/352/online+retail), автор Daqing Chen, 2015, DOI [10.24432/C5BW33](https://doi.org/10.24432/C5BW33). Це історія операцій британського інтернет-магазину. Вхідний файл `data/Online Retail.xlsx` зчитано функцією `pandas.read_excel`.

Період очищених даних — **{df.InvoiceDate.min():%d.%m.%Y}–{df.InvoiceDate.max():%d.%m.%Y}**. Один запис відповідає товарній позиції замовлення; один рахунок може містити декілька записів.

Для розрахунків використано `CustomerID` — клієнт, `InvoiceNo` — номер замовлення, `InvoiceDate` — дата, `Quantity` — кількість товару та `UnitPrice` — ціна у GBP. Ідентифікатор клієнта застосовується для групування записів і не входить до ознак K-means.

### Очищення

Дати й числові значення приведено до відповідних типів. Рядки з пропусками в обов’язкових полях вилучено: без ідентифікатора неможливо визначити клієнта, а без дати чи суми — коректно обчислити RFM. Скасування, позначені літерою `C` на початку номера рахунку, і операції з непозитивною кількістю або ціною виключено. Вартість товарної позиції розраховано як `TotalPrice = Quantity × UnitPrice`.

| Показник | Значення |
|---|---:|
| Початкова кількість рядків | {data_stats['raw_rows']:,} |
| Рядки без CustomerID | {data_stats['missing_customer_id']:,} |
| Скасування після видалення пропусків | {data_stats['cancellation_rows']:,} |
| Непозитивні кількості або ціни після попередніх фільтрів | {data_stats['nonpositive_quantity_or_price']:,} |
| Рядки після очищення | {data_stats['clean_rows']:,} |
| Унікальні клієнти | {len(rfm):,} |
| Унікальні замовлення | {data_stats['unique_invoices']:,} |

### Вибір ознак та обробка екстремальних значень

Для кожного клієнта сформовано три числові ознаки:

| Ознака | Зміст і розрахунок |
|---|---|
| Recency | Кількість днів між датою відліку та останньою покупкою |
| Frequency | Кількість унікальних InvoiceNo клієнта |
| Monetary | Сума TotalPrice за всіма врахованими покупками клієнта |

Датою відліку обрано **{snapshot:%d.%m.%Y}**, наступний день після останньої покупки. Менший Recency означає недавню активність, більші Frequency і Monetary — частіші покупки та вищі витрати. Frequency рахує замовлення, а не товарні рядки.

Коректні великі покупки не видалялися автоматично: вони можуть належати оптовим клієнтам. Для послаблення впливу екстремальних значень до кожної RFM-ознаки застосовано `log1p(x)`, тобто натуральний логарифм `1 + x`. Це зменшує асиметрію, але не усуває повністю чутливість K-means до викидів.

### Нормалізація

Після логарифмування застосовано `StandardScaler`: від кожного значення віднято середнє відповідного стовпця й поділено на його стандартне відхилення. Ознаки отримали порівнянний масштаб, тому сума покупок у фунтах не домінує в евклідових відстанях лише через великі числові значення.

## 2 Розвідувальний аналіз даних

Після агрегації отримано таблицю з {len(rfm):,} клієнтів і трьох ознак. Описова статистика наведена в початкових одиницях виміру.

{summary.to_markdown(floatfmt='.2f')}

Середні Frequency і Monetary перевищують відповідні медіани: розподіли зміщені в бік великих значень. Це підтверджує доцільність логарифмування перед кластеризацією.

![Гістограми RFM](01_rfm_distributions.png)

*Рисунок 1 — Розподіли RFM-ознак у логарифмічному масштабі.*

Для дослідження зв’язків побудовано діаграми розсіювання для всіх трьох пар RFM-ознак. Кожна точка відповідає клієнту; на осях відкладено логарифмовані значення.

![Зв’язки між RFM ознаками](02_rfm_relationships.png)

*Рисунок 2 — Попарні зв’язки між Recency, Frequency та Monetary.*

Кореляція Пірсона між логарифмованими Frequency і Monetary становить **{correlation.loc['Frequency','Monetary']:.3f}**: частіші покупки пов’язані з більшою загальною сумою витрат. Кореляції Recency із Frequency та Monetary становлять **{correlation.loc['Recency','Frequency']:.3f}** і **{correlation.loc['Recency','Monetary']:.3f}**. У цій вибірці клієнти з недавньою покупкою загалом мають більше замовлень і вищі витрати. Ці залежності описують дані та не доводять причинного зв’язку.

## 3 Реалізація K-means та вибір кількості кластерів

[K-means](https://scikit-learn.org/stable/modules/clustering.html) об’єднує клієнтів за близькістю їхніх стандартизованих RFM-векторів. Алгоритм почергово призначає точки найближчим центрам і перераховує центри груп, зменшуючи суму квадратів відстаней усередині кластерів.

Порівняно `k` від 2 до 8. Для кожного варіанта використано `random_state=42` та `n_init=10`. Фіксація випадковості забезпечує відтворюваність у тому самому середовищі; десять ініціалізацій зменшують залежність від початкових центрів.

Метод Elbow аналізує inertia — суму квадратів відстаней до центрів кластерів. Вона знижується зі збільшенням `k`, тому для автоматичного вибору використано найбільший Silhouette Score. Inertia наведено як додаткову характеристику.

{evaluation.to_markdown(index=False,floatfmt='.4f')}

![Порівняння кількості кластерів](03_k_selection.png)

*Рисунок 3 — Inertia та Silhouette Score для k від 2 до 8.*

Найбільше значення Silhouette отримано для **k = {best}**. Остаточну модель навчено на всіх {len(rfm):,} клієнтах. Номери кластерів є технічними мітками; назви сегментів призначено після аналізу медіан RFM.

## 4 Оцінювання та інтерпретація кластерів

[Silhouette Score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html) порівнює середню відстань клієнта до своєї групи із середньою відстанню до найближчої іншої групи. Значення лежить від −1 до 1; вищі значення означають краще відокремлення, а близькі до нуля — перекриття груп. Для швидкодії оцінку розраховано на фіксованій випадковій підвибірці з {min(2500,len(rfm)):,} клієнтів.

Підсумковий Silhouette Score — **{score.loc[score.k==best,'silhouette'].iloc[0]:.4f}**. Це найкращий результат серед перевірених варіантів, але він не означає повного відокремлення груп або відсотка точності класифікації. Для інтерпретації використано початкові, а не стандартизовані значення RFM. Медіани зменшують вплив окремих великих покупців на опис типового клієнта.

{comparison.to_markdown(floatfmt='.2f')}

{chr(10).join(chr(10)+paragraph+chr(10) for paragraph in cluster_text)}

Кластер {strongest_id} описує загалом активніших і цінніших покупців, а кластер {weakest_id} — покупців із нижчою залученістю. Назва «ризик втрати» є інтерпретацією профілю, а не результатом окремої моделі прогнозування відтоку. Характеристики групи не є жорсткими правилами для кожного її учасника.

## 5 Візуалізація кластерів

Для відображення трьох ознак на площині застосовано PCA. Перша компонента пояснює **{pca_ratio[0]:.2%}**, друга — **{pca_ratio[1]:.2%}** дисперсії перетворених даних; разом — **{sum(pca_ratio):.2%}**. PCA використовується тільки для графіка: K-means працює з трьома підготовленими RFM-ознаками.

![Кластери у просторі PCA](04_pca_clusters.png)

*Рисунок 4 — Проєкція клієнтів на дві головні компоненти з кольорами кластерів.*

На діаграмі показано взаємне розташування груп і клієнтів біля їхньої межі. Координати PCA є комбінаціями ознак, тому не вимірюються безпосередньо у днях або фунтах.

![Розміри кластерів та частки виручки](05_cluster_sizes_revenue.png)

*Рисунок 5 — Кількість клієнтів і частка врахованої виручки за кластерами.*

Порівняння розміру груп із часткою виручки показує концентрацію покупок: кластер {strongest_id} містить {strongest.Share_percent:.2f}% клієнтів і забезпечує {strongest.Revenue_share_percent:.2f}% врахованої суми продажів.

![Теплова карта профілів](06_cluster_heatmap.png)

*Рисунок 6 — Медіанні профілі кластерів після логарифмування та стандартизації між групами.*

Теплова карта порівнює відносні значення ознак. Вищий Recency означає давнішу останню покупку, тоді як вищі Frequency та Monetary означають більше замовлень і витрат. Колір не є універсальною оцінкою «краще» або «гірше».

## Обмеження

Повернення й скасування виключені з аналізу позитивних покупок, а не відняті від початкових продажів. Тому грошові показники не є чистою виручкою після повернень. Покупки без CustomerID не враховано.

Дані належать до 2010–2011 років і містять оптових покупців. Логарифмування послаблює вплив великих значень, проте результат залежить від вибраних ознак, періоду спостереження та припущень K-means. Найкращий силует серед k=2..8 не гарантує найкращої кількості сегментів для будь-якої бізнес-задачі.

## Висновки

Виконано всі основні етапи сегментації: завантаження та очищення транзакцій, вибір RFM-ознак, послаблення впливу екстремальних значень і стандартизацію, EDA, підбір кількості кластерів, навчання K-means, оцінювання та візуалізацію.

Із {data_stats['raw_rows']:,} вихідних товарних рядків залишилося {data_stats['clean_rows']:,}; для аналізу сформовано профілі {len(rfm):,} клієнтів. За максимальним Silhouette Score обрано {best} групи. Основна відмінність між ними полягає в давності покупок, частоті замовлень і сумі витрат.

Ключовий бізнес-висновок — значна концентрація продажів у групі активніших клієнтів: {strongest.Share_percent:.2f}% клієнтської бази формують {strongest.Revenue_share_percent:.2f}% врахованої виручки. Сегментація описує поведінкові відмінності та дає основу для окремого аналізу цих груп.

## Джерела

1. Chen, D. (2015). [Online Retail](https://doi.org/10.24432/C5BW33). UCI Machine Learning Repository. Датасет поширюється за [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Вихідний Excel включено без змін; очищення та RFM є похідними результатами цього проєкту.
2. scikit-learn. [Clustering and K means](https://scikit-learn.org/stable/modules/clustering.html).
3. scikit-learn. [Silhouette Score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html).

## Файли роботи

`src/segmentation.py` — реалізація; `requirements.txt` — залежності; `customer_segments.csv` — RFM і сегмент кожного клієнта; `cluster_profiles.csv` — характеристики груп; `k_evaluation.csv` — метрики вибору k; PNG — шість графіків. Шляхи до вихідного коду наведено від кореня папки CustomerSegmentation, таблиці та графіки розміщено в results.
"""
    (out/'REPORT_UA.md').write_text(report,encoding='utf-8')


def main():
    out=ROOT/'results';out.mkdir(exist_ok=True)
    data=load_data()
    clean,stats=clean_transactions(data)
    rfm,snapshot=calculate_rfm(clean)
    x,_=prepare_features(rfm)
    best,scores=pick_k(x,range(2,min(9,len(x))))
    model=KMeans(n_clusters=best,random_state=42,n_init=10)
    rfm['Cluster']=model.fit_predict(x)
    profiles=profile_clusters(rfm)
    names=name_clusters(profiles)
    profiles['Segment_name']=[names[int(c)] for c in profiles.index]
    rfm['Segment_name']=rfm.Cluster.map(names)
    rfm.to_csv(out/'customer_segments.csv',encoding='utf-8-sig')
    profiles.to_csv(out/'cluster_profiles.csv',encoding='utf-8-sig')
    scores.to_csv(out/'k_evaluation.csv',index=False)
    ratios=charts(rfm,x,scores,profiles,out)
    build_report(stats,clean,rfm,profiles,scores,best,snapshot,ratios,out)
    print('Готово! Результати в:',out.resolve())
    print('Вибране k:',best,'\n',profiles.to_string())

if __name__=='__main__': main()
